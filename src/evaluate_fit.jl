export objective, error_metric, impute, impute_missing

### OBJECTIVE FUNCTION EVALUATION FOR MPCA
function objective(glrm::GLRM, X::Array{Float64,2}, Y::Array{Float64,2},
                   XY::Array{Float64,2};
                   yidxs = get_yidxs(glrm.losses), # mapping from columns of A to columns of Y; by default, the identity
                   include_regularization=true)
    m,n = size(glrm.A)
    @assert(size(XY)==(m,yidxs[end][end]))
    @assert(size(Y)==(glrm.k,yidxs[end][end]))
    @assert(size(X)==(glrm.k,m))
    err = 0.0
    for j=1:n
        for i in glrm.observed_examples[j]
            err += evaluate(glrm.losses[j], XY[i,yidxs[j]], glrm.A[i,j])
        end
    end
    # add regularization penalty
    if include_regularization
        err += calc_penalty(glrm,X,Y; yidxs = yidxs)
    end
    return err
end

function objective(fglrm::FairGLRM, X::Array{Float64,2}, Y::Array{Float64,2},
                   XY::Array{Float64,2};
                   yidxs = get_yidxs(fglrm.losses), # mapping from columns of A to columns of Y; by default, the identity
                   include_regularization=true)
    m,n = size(fglrm.A)
    @assert(size(XY)==(m,yidxs[end][end]))
    @assert(size(Y)==(fglrm.k,yidxs[end][end]))
    @assert(size(X)==(fglrm.k,m))
    # Use the provided group functional to evaluate the total loss
    total_err = evaluate(fglrm.group_functional, fglrm.losses, XY, fglrm.A, fglrm.Z, fglrm.observed_features)
    if eltype(fglrm.observed_features) == UnitRange{Int64}
        magnitude_Ω = sum(length(f) for f in fglrm.observed_features)
    else
        magnitude_Ω = size(fglrm.observed_features, 1) * size(fglrm.observed_features, 2)
    end
    total_err *= magnitude_Ω
    # add regularization penalty
    if include_regularization
        total_err += calc_penalty(fglrm,X,Y; yidxs = yidxs)
    end
    return total_err
end

function row_objective(glrm::AbstractGLRM, i::Int, x::AbstractArray, Y::Array{Float64,2} = glrm.Y;
                   yidxs = get_yidxs(glrm.losses), # mapping from columns of A to columns of Y; by default, the identity
                   include_regularization=true)
    m,n = size(glrm.A)
    err = 0.0
    XY = x'*Y
    for j in glrm.observed_features[i]
        err += evaluate(glrm.losses[j], XY[1,yidxs[j]], glrm.A[i,j])
    end
    # add regularization penalty
    if include_regularization
        err += evaluate(glrm.rx[i], x)
    end
    return err
end

function row_objective(fglrm::FairGLRM, i::Int, x::AbstractArray, Y::Array{Float64,2} = fglrm.Y;
                       yidxs = get_yidxs(fglrm.losses), # mapping from columns of A to columns of Y; by default, the identity
                       include_regularization=true)
    XY = x'*Y
    # Use the provided group functional to evaluate the total loss
    xy = []
    for yidx in yidxs
        push!(xy, XY[yidx])
    end
    err = evaluate(fglrm.group_functional, fglrm.losses, xy, fglrm.A[i, :], [Set(1)], fglrm.observed_features[i], yidxs=yidxs)
    # add regularization penalty
    if include_regularization
        err += evaluate(fglrm.rx[i], x)
    end
    return err
end

function col_objective(glrm::AbstractGLRM, j::Int, y::AbstractArray, X::Array{Float64,2} = glrm.X;
                   include_regularization=true)
    m,n = size(glrm.A)
    sz = size(y)
    if length(sz) == 1 colind = 1 else colind = 1:sz[2] end
    err = 0.0
    XY = X'*y
    obsex = glrm.observed_examples[j]
    @inbounds XYj = XY[obsex,colind]
    @inbounds Aj = convert(Array, glrm.A[obsex,j])
    err += evaluate(glrm.losses[j], XYj, Aj)
    # add regularization penalty
    if include_regularization
        err += evaluate(glrm.ry[j], y)
    end
    return err
end

function col_objective(fglrm::FairGLRM, j::Int, y::AbstractArray, X::Array{Float64,2} = fglrm.X;
                       include_regularization=true)
    sz = size(y)
    if length(sz) == 1 colind = 1 else colind = 1:sz[2] end
    XY = X'*y
    obsex = fglrm.observed_examples[j]
    obsset = Set(obsex)
    # Want to use a subset of Z, including only the elements that are observed
    groups = [intersect(g, obsset) for g in fglrm.Z]
    @inbounds XYj = XY[obsex,colind]
    @inbounds Aj = convert(Array, fglrm.A[obsex,j])
    if length(sz) == 1 feature_dims = 1 else feature_dims = length(colind) end
    err = evaluate(fglrm.group_functional, [fglrm.losses[j]], XYj, Aj, groups, ones(Int64, feature_dims, length(obsex)), yidxs=[colind])
    # add regularization penalty
    if include_regularization
        err += evaluate(fglrm.ry[j], y)
    end
    return err
end

# The user can also pass in X and Y and `objective` will compute XY for them
function objective(glrm::GLRM, X::Array{Float64,2}, Y::Array{Float64,2};
                   sparse=false, include_regularization=true,
                   yidxs = get_yidxs(glrm.losses), kwargs...)
    @assert(size(Y)==(glrm.k,yidxs[end][end]))
    @assert(size(X)==(glrm.k,size(glrm.A,1)))
    XY = Array{Float64}(undef, (size(X,2), size(Y,2)))
    if sparse
        # Calculate X'*Y only at observed entries of A
        m,n = size(glrm.A)
        err = 0.0
        for j=1:n
            for i in glrm.observed_examples[j]
                err += evaluate(glrm.losses[j], dot(X[:,i],Y[:,yidxs[j]]), glrm.A[i,j])
            end
        end
        if include_regularization
            err += calc_penalty(glrm,X,Y; yidxs = yidxs)
        end
        return err
    else
        # dense calculation variant (calculate XY up front)
        gemm!('T','N',1.0,X,Y,0.0,XY)
        return objective(glrm, X, Y, XY; include_regularization=include_regularization, yidxs = yidxs, kwargs...)
    end
end
# Or just the GLRM and `objective` will use glrm.X and .Y
objective(glrm::GLRM; kwargs...) = objective(glrm, glrm.X, glrm.Y; kwargs...)
objective(fglrm::FairGLRM; kwargs...) = objective(fglrm, fglrm.X, fglrm.Y; kwargs...)

# For shared arrays
# TODO: compute objective in parallel
objective(glrm::ShareGLRM, X::SharedArray{Float64,2}, Y::SharedArray{Float64,2}) =
    objective(glrm, X.s, Y.s)

# Helper function to calculate the regularization penalty for X and Y
function calc_penalty(glrm::AbstractGLRM, X::Array{Float64,2}, Y::Array{Float64,2};
    yidxs = get_yidxs(glrm.losses))
    m,n = size(glrm.A)
    @assert(size(Y)==(glrm.k,yidxs[end][end]))
    @assert(size(X)==(glrm.k,m))
    penalty = 0.0
    for i=1:m
        penalty += evaluate(glrm.rx[i], view(X,:,i))
    end
    for f=1:n
        penalty += evaluate(glrm.ry[f], view(Y,:,yidxs[f]))
    end
    return penalty
end

function calc_penalty(glrm::FairGLRM, X::Array{Float64, 2}, Y::Array{Float64, 2})
    yidxs = get_yidxs(glrm.losses)
    m,n = size(glrm.A)
    @assert(size(Y)==(glrm.k,yidxs[end][end]))
    @assert(size(X)==(glrm.k,m))
    penalty = 0.0
    for i=1:m
        penalty += evaluate(glrm.rx[i], view(X,:,i))
    end
    for f=1:n
        penalty += evaluate(glrm.ry[f], view(Y,:,yidxs[f]))
    end
    for k_prime=1:glrm.k
        penalty += evaluate(glrm.rkx[k_prime], view(X, k_prime, :))
        penalty += evaluate(glrm.rky[k_prime], view(Y, k_prime, :))
    end
    return penalty
end

## ERROR METRIC EVALUATION (BASED ON DOMAINS OF THE DATA)
function raw_error_metric(glrm::AbstractGLRM, XY::Array{Float64,2}, domains::Array{Domain,1};
    yidxs = get_yidxs(glrm.losses))
    m,n = size(glrm.A)
    err = 0.0
    for j=1:n
        for i in glrm.observed_examples[j]
            err += error_metric(domains[j], glrm.losses[j], XY[i,yidxs[j]], glrm.A[i,j])
        end
    end
    return err
end
function std_error_metric(glrm::AbstractGLRM, XY::Array{Float64,2}, domains::Array{Domain,1};
    yidxs = get_yidxs(glrm.losses))
    m,n = size(glrm.A)
    err = 0.0
    for j=1:n
        column_mean = 0.0
        column_err = 0.0
        for i in glrm.observed_examples[j]
            column_mean += glrm.A[i,j]^2
            column_err += error_metric(domains[j], glrm.losses[j], XY[i,yidxs[j]], glrm.A[i,j])
        end
        column_mean = column_mean/length(glrm.observed_examples[j])
        if column_mean != 0
            column_err = column_err/column_mean
        end
        err += column_err
    end
    return err
end
function error_metric(glrm::AbstractGLRM, XY::Array{Float64,2}, domains::Array{Domain,1};
    standardize=false,
    yidxs = get_yidxs(glrm.losses))
    m,n = size(glrm.A)
    @assert(size(XY)==(m,yidxs[end][end]))
    if standardize
        return std_error_metric(glrm, XY, domains; yidxs = yidxs)
    else
        return raw_error_metric(glrm, XY, domains; yidxs = yidxs)
    end
end
# The user can also pass in X and Y and `error_metric` will compute XY for them
function error_metric(glrm::AbstractGLRM, X::Array{Float64,2}, Y::Array{Float64,2}, domains::Array{Domain,1}=Domain[l.domain for l in glrm.losses]; kwargs...)
    XY = Array{Float64}(undef,(size(X,2), size(Y,2)))
    gemm!('T','N',1.0,X,Y,0.0,XY)
    error_metric(glrm, XY, domains; kwargs...)
end
# Or just the GLRM and `error_metric` will use glrm.X and .Y
error_metric(glrm::AbstractGLRM, domains::Array{Domain,1}; kwargs...) = error_metric(glrm, glrm.X, glrm.Y, domains; kwargs...)
error_metric(glrm::AbstractGLRM; kwargs...) = error_metric(glrm, Domain[l.domain for l in glrm.losses]; kwargs...)

# Use impute and errors over GLRMS
impute(glrm::AbstractGLRM) = impute(glrm.losses, glrm.X'*glrm.Y)
function impute_missing(glrm::AbstractGLRM)
  Ahat = impute(glrm)
  for j in 1:size(glrm.A,2)
    for i in glrm.observed_examples[j]
      Ahat[i,j] = glrm.A[i,j]
    end
  end
  return Ahat
end
