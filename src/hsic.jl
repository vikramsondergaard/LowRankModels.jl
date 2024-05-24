import Distributions: Gamma

export evaluate, grad, get_independence_criterion, get_nfsic, HSIC, NFSIC

"""
Code for computing HSIC of two matrices

Translated from Python code at: https://github.com/amber0309/HSIC/blob/master/HSIC.py
"""

"""
Julia implementation of Hilbert Schmidt Independence Criterion
hsic_gam implements the HSIC test using a Gamma approximation

Gretton, A., Fukumizu, K., Teo, C. H., Song, L., Scholkopf, B., 
& Smola, A. J. (2007). A kernel statistical test of independence. 
In Advances in neural information processing systems (pp. 585-592).

Shoubo (shoubo.sub AT gmail.com)
09/11/2016

Inputs:
X 		n by dim_x matrix
Y 		n by dim_y matrix
alph 		level of test

Outputs:
testStat	test statistics
thresh		test threshold for level alpha test
"""

abstract type IndependenceCriterion end

function get_independence_criterion(Y::AbstractArray{Float32, 2}, X::AbstractArray{Float32, 2}, ic::DataType)
    if ic == HSIC
        return get_hsic(Y)
    elseif ic == NFSIC
        return get_nfsic(Y, X)
    else
        error("Independence criterion $ic not implemented yet!")
    end
end
get_independence_criterion(Y::AbstractArray{Float64, 2}, ic::DataType) =
    get_independence_criterion(Float32.(Y), ic)
get_independence_criterion(Y::AbstractArray{Float64, 1}, ic::DataType) =
    get_independence_criterion(Float32.(Y)[:, :], ic)
get_independence_criterion(Y::AbstractArray{Float64, 1}, X::AbstractArray{Float64, 1}, ic::DataType) =
    get_independence_criterion(Float32.(Y)[:, :], Float32.(X)[:, :], ic)
get_independence_criterion(Y::AbstractArray{Float64, 1}, X::AbstractArray{Float32, 1}, ic::DataType) =
    get_independence_criterion(Float32.(Y)[:, :], X[:, :], ic)
get_independence_criterion(Y::AbstractArray{Float64, 2}, X::AbstractArray{Float64, 2}, ic::DataType) =
    get_independence_criterion(Float32.(Y), Float32.(X), ic)
get_independence_criterion(Y::AbstractArray{Float64, 2}, X::AbstractArray{Float32, 1}, ic::DataType) =
    get_independence_criterion(Float32.(Y), X[:, :], ic)
get_independence_criterion(Y::AbstractArray{Float64, 2}, X::AbstractArray{Float64, 1}, ic::DataType) =
    get_independence_criterion(Float32.(Y), Float32.(X)[:, :], ic)

eye(n::Int64) = CuMatrix(I, n, n)

struct HSIC <: IndependenceCriterion
    HLH::AbstractArray{Float32, 2}
end
function get_hsic(Y::AbstractArray{Float32})
    n = size(Y, 1)
    H = eye(n) .- CUDA.ones(Float32, n, n) ./ n
    width_y = get_width(Y)
    L = rbf_dot(Y, Y, width_y)
    HLH = H * L * H
    HSIC(HLH)
end
get_hsic(Y::AbstractArray{Float64}) = get_hsic(Float32.(Y))

function rbf_dot(pattern1::AbstractArray, pattern2::AbstractArray, deg::Float64)
    """
    Constructs the RBF kernel matrix between two input sources of data,
    `pattern1` and `pattern2`, using a given Gaussian kernel width.

    ## Parameters
    - pattern1: the first input source of data
    - pattern2: the second input source of data
    - deg: the width of the Gaussian kernel to calculate

    ## Returns
    A kernel matrix H calculating the distance between each point in pattern1
    and pattern2.

    https://github.com/amber0309/HSIC/blob/master/HSIC.py
    """
    # Get sum of Hadamard product along the rows
    # println("Cal")
    G = sum(pattern1 .* pattern1, dims=2)
    H = sum(pattern2 .* pattern2, dims=2)

    # Project both matrices into a common space
    K = (G .+ H') .- (2 .* (pattern1 * pattern2'))

    # Apply Gaussian distance calculation
    exp.(-K ./ (2 * deg^2))
end

rbf_dot(pattern1::AbstractArray, pattern2::AbstractArray, deg::Float32) = rbf_dot(pattern1, pattern2, Float64(deg))
rbf_dot(pattern1::AbstractArray, pattern2::AbstractArray, deg::Int64) = rbf_dot(pattern1, pattern2, Float64(deg))

function rbf_dot(X::AbstractArray, deg²::Float64)
    """
    A convenience function to calculate the kernel matrix where pattern1 and
    pattern2 are the same.
    """
    G = mapreduce(identity, +, X .* X; dims=2)
    H = (G .+ G') .- (2 .* (X * X'))
    map(exp, -H ./ (2 * deg²))
end

function get_width(M::AbstractArray)
    """
    Find the best width for the RBF kernel. May be deprecated because of the
    optimisation procedure I'm developing for NFSIC, which also handles
    Gaussian kernel widths.

    https://github.com/amber0309/HSIC/blob/master/HSIC.py
    """
    G = mapreduce(identity, +, M .* M; dims=2)

    dists = (G .+ G') .- (2 .* (M * M'))
    dists .-= tril(dists)
    dists = filter(d -> d > 0, dists)

    if isempty(dists)
        return 0
    else
        return sqrt(0.5 * median(dists))
    end
end

function evaluate(hsic::HSIC, X::AbstractArray{Float32};
        alph::Float64=0.5)
    """
    Evaluates the Hilbert Schmidt Independence Criterion for two sources of
    data, X and Y.

    ## Parameters
    - hsic: a HSIC instance which stores Y. This is done for domain-specific
      performance purposes (in my use case, Y never changes, so the program
      runs ~2x faster by just storing Y in a struct).
    - X: the other source of input data apart from Y.
    - alph: the test statistic, not used in this use case but taken from the
      original source code.

    ## Returns
    The Hilbert-Schmidt Independence Criterion test score between X and Y. The
    larger the value, the more statistically dependent X and Y are. This value
    is always non-negative.

    https://github.com/amber0309/HSIC/blob/master/HSIC.py
    """
    n = size(X, 1)
    # if n == 1 return 0 end
    # width_x = get_width(X)
    # if width_x == 0 return 0 end

    K = rbf_dot(X, Float64(var(X)))
    hsic_mat = K * hsic.HLH

    test_stat = reduce(+, diag(hsic_mat))

    test_stat / n^2
end

# function hsic_gam!(hsic::HSIC, X::AbstractArray, e::Int)
#     # Get σ to get the width of the newly changed distribution
#     n = size(X, 1)
#     width_x = get_width(X)
#     # The new RBF only differs from the old RBF for the changed value e: so we
#     # only need a vector, not a matrix
#     K = rbf_dot(X, X, width_x)
#     # Old RBF needs to be taken for calculating difference
#     old_rbf = hsic.K[e, :]
#     new_rbf = K[e, :]
#     rbf_diff = new_rbf - old_rbf
#     a = CuArray([i == e for i=1:n])
#     A = broadcast(|, a, a')
#     HLH = hsic.HLH .* A
#     # These parts of the trace are the only differences from the old HSIC
#     # Note: this has a double-up at (e, e)
#     trace = reduce(+, broadcast(*, HLH, rbf_diff)) / n^2
#     # Update the trace value so I can get it back later
#     hsic.hsic += trace
#     hsic.X = X
#     hsic.K = K
#     hsic.hsic
# end

function grad(hsic::HSIC, X::AbstractArray)
    """
    Calculate the gradient of the Hilbert-Schmidt Independence Criterion using
    two given input sources of data.

    ## Parameters
    - hsic: a HSIC instances which stores Y. See the documentation for
      evaluate() (just above this) for more details.
    - X: the other source of input data, apart from Y.

    ## Returns
    The gradient of the Hilbert-Schmidt Independence Criterion.
    """
    n = size(X, 1)
    if n == 1 return zeros(n) end
    dim_x = size(X, 2)
    width_x = var(X)
    if width_x == 0 return zeros(n) end

    Km = rbf_dot(X, X, Float64(width_x)) .* broadcast(-, X, X') # Hadamard product
    Kc = Km * hsic.HLH

    trace = 2 * reduce(+, diag(Kc)) / n^2 / width_x^2
    return trace
end

# function hsic_grad!(G::AbstractArray, X::AbstractArray, Y::AbstractArray)
#     n = size(X, 1)
#     if n == 1 return nothing end
#     dim_x = size(X, 2)
#     width_x = get_width(X)
#     width_y = get_width(Y)
#     if width_x == 0 || width_y == 0 return nothing end
    
#     H = CuMatrix(I, n, n) - CUDA.ones(Float64, n, n) ./ n

#     K = rbf_dot(X, X, width_x)
#     L = rbf_dot(Y, Y, width_y)

#     M = broadcast(-, X, X')

#     Lc = (H * L) * H
#     Km = K .* M # Hadamard product
#     Kc = (H * Km) * H
#     G .+= 2 * sum(Kc' .* Lc) / n / width_x^2
#     return nothing
# end

######### RANDOM FOURIER FEATURE APPROXIMATION #############

# Code largely taken from Gregory Gunderson's code:
# https://github.com/gwgundersen/random-fourier-features/blob/master/kernelapprox.py

# But translated into Julia and adapted for CUDA

function hsic_rff(hsic::HSIC, X::AbstractArray; n_samples::Int64=floor(Int, length(X)/10))
    n = size(X, 1)
    dim_x = size(X, 2)
    W = CUDA.randn(n_samples, dim_x)
    b = CUDA.rand(n_samples) * 2 * pi
    WXᵀ = W * X'
    Z = sqrt(2 / n_samples) .* cos.(WXᵀ .+ b)
    K = Z' * Z
    hsic_mat = K * hsic.HLH

    test_stat = reduce(+, diag(hsic_mat))

    test_stat / n^2
end

struct NFSIC <: IndependenceCriterion
    L::T where T<:AbstractArray{Float32, 2}
    mean_l::T where T<:AbstractArray{Float32, 2}
    Lt::T where T<:AbstractArray{Float32, 2}
    V::AbstractArray
    W::AbstractArray
    width_x::Float64
    width_y::Float64
end
function get_nfsic(Y::AbstractArray)
    # width_y = get_width(Y)
    idxs = rand(1:size(Y, 1), 1250)
    W = CuArray(Y[idxs, :])
    get_nfsic(Y, W, Float64(var(Y)))
end
function get_nfsic(Y::AbstractArray, X::AbstractArray)
    V, W, width_x, width_y = optimize_locs_widths(X, Y, n_test_locs=2, max_iter=2000, tol_fun=1e-4)
    L = rbf_dot(Y, W, sqrt(width_y))
    mean_l = mean(L, dims=1)
    NFSIC(L, mean_l, L .- mean_l, V, W, sqrt(width_x), sqrt(width_y))
end
function get_nfsic(Y::AbstractArray, V::AbstractArray, W::AbstractArray, width_x::Float32, width_y::Float32)
    L = rbf_dot(Y, W, width_y)
    mean_l = mean(L, dims=1)
    NFSIC(L, mean_l, L .- mean_l, V, W, width_x, width_y)
end
# get_nfsic(Y::AbstractArray{Float64}, W::AbstractArray{Float32}) = get_nfsic(Float32.(Y), W)
# get_nfsic(Y::AbstractArray{Float32}, W::AbstractArray{Float64}) = get_nfsic(Y, Float32.(W))
# get_nfsic(Y::AbstractArray{Float64}, W::AbstractArray{Float64}) = get_nfsic(Float32.(Y), Float32.(W))

# function evaluate(nfsic::NFSIC, X::AbstractArray{Float32}; optim=false, reg=1)
#     n = size(X, 1)
#     idxs = rand(1:n, 1250)
#     V = CuArray(X[idxs, :])
#     evaluate(nfsic, X, V)
# end

function evaluate(nfsic::NFSIC, X::AbstractArray{Float64}; reg=1)
    evaluate(nfsic, Float32.(X), reg=reg)
end

function evaluate(nfsic::NFSIC, X::AbstractArray{Float32}; reg=1)
    n = size(X, 1)
    J = size(nfsic.V, 1)
    # width_x = get_width(X)
    
    K = rbf_dot(X, nfsic.V, nfsic.width_x) # n x J
    L = nfsic.L                            # n x J
    
    # mean
    mean_k = mean(K; dims=1)
    mean_l = nfsic.mean_l

    # biased
    u = CuArray(mean(K .* L; dims=1) - (mean_k .* mean_l))
    # cov
    # Generic covariance
    Kt = K .- mean_k
    Lt = nfsic.Lt
    
    Snd_mo = Kt .* Lt
    Sig = (Snd_mo' * Snd_mo ./ n) - (u' * u)
    nfsic_from_u_sig(Float32.(u), Float32.(Sig), n, reg) * (n / J)
end

function evaluate(nfsic::NFSIC, X::AbstractArray{Float32}, groups::AbstractArray{Int64, 1}; reg=1)
    L = CuArray(Array(nfsic.L)[groups, :])
    mean_l = mean(L, dims=1)
    V_W_groups = filter(g -> g <= size(V, 1), groups)
    V = nfsic.V[V_W_groups, :]
    W = nfsic.W[V_W_groups, :]
    new_nfsic = NFSIC(L, mean_l, L .- mean_l, V, W, nfsic.width_x, nfsic.width_y)
    return evaluate(new_nfsic, X, reg=reg)
end

function nfsic_from_u_sig(u::AbstractArray{Float32, 2}, Sig::AbstractArray{Float32, 2}, n::Int64, reg::Int=0)
    J = length(u)
    # println("J is $J")
    # println("The size of u is $(size(u))")
    # show(u); println()
    # show(Sig); println()
    if J == 1
        r = max(reg, 1250)
        u_dot_one = reduce(+, u)^2
        sig_dot_one = reduce(+, Sig)
        s = Float64(n) * u_dot_one / (r + sig_dot_one)
    else
        if reg <= 0
            try
                sol = Sig \ u'
                # println("sol is:"); show(sol); println()
                s = n * dot(sol, u)
            catch e
                throw(e)
                try
                    # singular matrix
                    # eigen decompose
                    evals = eigvals(Sig); eV = eigvecs(Sig)
                    evals = max.(0, evals)
                    # find the non-zero second smallest eigenvalue
                    snd_small = minimum(filter(e -> e > 0, evals))
                    evals = max.(snd_small, evals)

                    # reconstruct Sig
                    Sig = dot.(eV, diag(evals)) * eV'
                    # try again
                    sol = Sig \ u
                    println("sol is:")
                    show(sol)
                    s = n * dot(sol, u)
                catch e
                    throw(e)
                    s = 0
                end
            end
        else
            # assume reg is a number
            s = n * dot(((Sig + reg .* eye(size(Sig, 1))) \ u'), u)
        end
    end
    return s
end

nfsic_from_u_sig(u::AbstractArray{Float64, 2}, Sig::AbstractArray{Float64, 2}, n::Int64, reg::Int=0) =
    nfsic_from_u_sig(Float32.(u), Float32.(Sig), n, reg)

function grad(nfsic::NFSIC, X::AbstractArray{Float32}; optim=false, reg=1)
    n = size(X, 1)

    J = size(nfsic.V, 1)
    
    K = rbf_dot(X, nfsic.V, nfsic.width_x)              # n x J
    Km = -1 .* K .* (X .- nfsic.V') ./ nfsic.width_x^2  # gradient
    L = nfsic.L                                         # n x J
    
    # mean
    mean_k = mean(Km; dims=1)
    mean_l = nfsic.mean_l

    # biased
    u = CuArray(mean(Km .* L; dims=1) - (mean_k .* mean_l))
    # cov
    # Generic covariance
    Kt = Km .- mean_k
    Lt = nfsic.Lt
    
    Snd_mo = Kt .* Lt
    Sig = (Snd_mo' * Snd_mo ./ n) - (u' * u)
    nfsic_from_u_sig(Float32.(u), Float32.(Sig), n, reg)
end

MaybeFloat64 = Union{Float64, Nothing}
MaybeInt64 = Union{Int64, Nothing}

"""
Optimize the test locations V, W and the Gaussian kernel width by 
maximizing a test power criterion. X, Y should not be the same data as
used in the actual test (i.e., should be a held-out set). 

- max_iter: #gradient descent iterations
- batch_proportion: (0,1] value to be multipled with n giving the batch 
    size in stochastic gradient. 1 = full gradient ascent.
- tol_fun: termination tolerance of the objective value
- If the lb, ub bounds are None, use fraction of the median heuristics 
    to automatically set the bounds.
        
Return (V test_locs, W test_locs, gaussian width for x, gaussian width
    for y, info log)
"""

"""
Optimize the empirical version of Lambda(T) i.e., the criterion used 
to optimize the test locations, for the test based 
on difference of mean embeddings with Gaussian kernel. 
Also optimize the Gaussian width.

https://github.com/wittawatj/fsic-test/blob/master/fsic/indtest.py
"""
function optimize_locs_widths(X::AbstractArray, Y::AbstractArray;
        n_test_locs::Int=5, max_iter::Int=400, V_step::Float64=1., W_step::Float64=1.,
        gwidthx_step::Float64=1., gwidthy_step::Float64=1., batch_proportion::Float64=1.,
        tol_fun::Float64=1.0e-3, step_pow::Float64=0.5, seed::Int=1,
        reg::Float64=1.0e-5, gwidthx_lb::MaybeFloat64=nothing,
        gwidthy_lb::MaybeFloat64=nothing, gwidthx_ub::MaybeFloat64=nothing,
        gwidthy_ub::MaybeFloat64=nothing)
    J = n_test_locs
    # Use grid search to initialise the gwidths for both X, Y
    n_gwidth_cand = 5
    gwidth_factors = fill(2, n_gwidth_cand).^LinRange(-3, 3, n_gwidth_cand)
    filter!(w -> w > 0, gwidth_factors)
    medx2 = meddistance(X, 1000)^2
    medy2 = meddistance(Y, 1000)^2

    V, W = init_check_subset(X, Y, medx2 * 2, medy2 * 2, J, n_cand=500, subsample=size(X,1))
    best_widthx, best_widthy = nfsic_grid_search_kernel(X, Y, V, W,
        medx2 .* gwidth_factors, medy2 .* gwidth_factors)
    @assert typeof(best_widthx) <: Real "best_widthx not real. Was $best_widthx"
    @assert typeof(best_widthy) <: Real "best_widthy not real. Was $best_widthy"
    @assert best_widthx > 0 "best_widthx not positive. Was $best_widthx"
    @assert best_widthy > 0 "best_widthy not positive. Was $best_widthy"

    # set the width bounds
    fac_min = 5.0e-2
    fac_max = 5.0e3
    gwidthx_lb = gwidthx_lb == nothing ? fac_min * medx2 : gwidthx_lb
    gwidthy_lb = gwidthy_lb == nothing ? fac_min * medy2 : gwidthy_lb
    gwidthx_ub = gwidthx_ub == nothing ? fac_max * medx2 : gwidthx_lb
    gwidthy_ub = gwidthy_ub == nothing ? fac_max * medy2 : gwidthy_ub

    V, W, width_x, width_y = generic_optimize_locs_widths(X, Y, V, W,
        best_widthx, best_widthy, func_obj; 
        max_iter=max_iter, V_step=V_step, W_step=W_step,
        gwidthx_step=gwidthx_step, gwidthy_step=gwidthy_step,
        batch_proportion=batch_proportion, tol_fun=tol_fun, step_pow=step_pow,
        reg=reg, gwidthx_lb=gwidthx_lb, gwidthx_ub=gwidthx_ub,
        gwidthy_lb=gwidthy_lb, gwidthy_ub=gwidthy_ub)

    # make sure that the optimized gwidthx, gwidthy are not too small
    # or too large.
    fac_min = 5e-2
    fac_max = 5e3
    width_x = max(fac_min * medx2, 1e-7, min(fac_max * medx2, width_x))
    width_y = max(fac_min * medy2, 1e-7, min(fac_max * medy2, width_y))
    return (V, W, width_x, width_y)
end

"""
https://github.com/wittawatj/fsic-test/blob/master/fsic/util.py

Compute the median of pairwise distances (not distance squared) of points
    in the matrix.  Useful as a heuristic for setting Gaussian kernel's width.

## Parameters

X : n x d numpy array
    
mean_on_fail: True/False. If True, use the mean when the median distance is 0.
    This can happen especially, when the data are discrete e.g., 0/1, and 
    there are more slightly more 0 than 1. In this case, the m

## Return

median distance

"""
function meddistance(X::AbstractArray, subsample::Int64)
    @assert subsample > 0
    n = size(X, 1)
    ind = sample(1:n, min(subsample, n), replace=false)
    X_ind = X[ind, :]
    D = dist_matrix(X_ind, X_ind)
    tril!(D, -1)
    Tri = filter(d -> d > 0, D)
    med = median(Tri)
    return med <= 0 ? mean(Tri) : med
end

"""
    Construct a pairwise Euclidean distance matrix of size 
    size(X, 1) x size(Y, 1)
"""
function dist_matrix(X::AbstractArray, Y::AbstractArray)
    sx = mapreduce(x -> x^2, +, X, dims=2)
    sy = mapreduce(y -> y^2, +, Y, dims=2)
    D2 = sx .+ sy' - (2 .* (X * Y'))
    # max.() is used to prevent numerical errors from taking sqrt of negative
    # numbers
    D2 = sqrt.(max.(D2, 0))
    D2
end

"""
    Evaluate a set of locations to find the best locations to initialize. 
    The location candidates are drawn from the joint and the product of 
    the marginals.

    ## Parameters
    - subsample the data when computing the objective 
    - n_cand: number of times to draw from the joint and the product 
        of the marginals.
    
    ## Return    
    V, W
"""
function init_check_subset(X::AbstractArray, Y::AbstractArray, widthx::Float64,
        widthy::Float64, n_test_locs::Int64;
        n_cand::Int64=50, subsample::Int64=2000, seed::Int64=3)
    init_check_subset(X, Y, Float32(widthx), Float32(widthy), n_test_locs, n_cand=n_cand, subsample=subsample, seed=seed)
end
function init_check_subset(X::AbstractArray, Y::AbstractArray, widthx::Float32,
    widthy::Float64, n_test_locs::Int64;
    n_cand::Int64=50, subsample::Int64=2000, seed::Int64=3)
init_check_subset(X, Y, widthx, Float32(widthy), n_test_locs, n_cand=n_cand, subsample=subsample, seed=seed)
end
function init_check_subset(X::AbstractArray, Y::AbstractArray, widthx::Float32,
        widthy::Float32, n_test_locs::Int64;
        n_cand::Int64=50, subsample::Int64=2000, seed::Int64=3)
    n = size(X, 1)

    # from the joint
    best_obj_joint = 0
    best_V_joint::Union{AbstractArray, Nothing} = nothing
    best_W_joint::Union{AbstractArray, Nothing} = nothing
    for _=1:n_cand
        V, W = init_locs_joint_subset(X, Y, n_test_locs)
        if subsample < n
            I = sample(1:n, n_test_locs, ordered=true, replace=false)
            nfsic = get_nfsic(Y[I, :], V, W, widthx, widthy)
            obj_joint = evaluate(nfsic, X[I, :];reg=0)
        else
            nfsic = get_nfsic(Y, V, W, widthx, widthy)
            obj_joint = evaluate(nfsic, X; reg=0)
        end
        if obj_joint > best_obj_joint || best_V_joint == nothing
            best_V_joint = V
            best_W_joint = W
            best_obj_joint = obj_joint
        end
    end

    best_obj_prod = 0
    best_V_prod::Union{AbstractArray, Nothing} = nothing
    best_W_prod::Union{AbstractArray, Nothing} = nothing
    for _=1:n_cand
        V, W = init_locs_marginals_subset(X, Y, n_test_locs)
        if subsample < n
            I = sample(1:n, n_test_locs, ordered=true, replace=false)
            nfsic = get_nfsic(Y[I, :], V, W, widthx, widthy)
            obj_prod = evaluate(nfsic, X[I, :]; reg=0)
        else
            nfsic = get_nfsic(Y, V, W, widthx, widthy)
            obj_prod = evaluate(nfsic, X; reg=0)
        end
        if obj_prod > best_obj_prod || best_V_prod == nothing
            best_V_prod = V
            best_W_prod = W
            best_obj_prod = obj_prod
        end
    end

    return best_obj_joint >= best_obj_prod ? (best_V_joint, best_W_joint) : (best_V_prod, best_W_prod)
end

function init_locs_joint_subset(X::AbstractArray, Y::AbstractArray,
        n_test_locs::Int64)
    n = size(X, 1)
    I = sample(1:n, n_test_locs)
    V = X[I, :]
    W = Y[I, :]
    return V, W
end

function init_locs_marginals_subset(X::AbstractArray, Y::AbstractArray,
        n_test_locs::Int64)
    n = size(X, 1)
    Ix = sample(1:n, n_test_locs)
    Iy = sample(1:n, n_test_locs)
    V = X[Ix, :]
    W = Y[Iy, :]
    return V, W
end

function nfsic_grid_search_kernel(X::AbstractArray, Y::AbstractArray,
        V::AbstractArray, W::AbstractArray, list_gwidthx::AbstractArray,
        list_gwidthy::AbstractArray)
    n = size(X, 1)
    J = size(V, 1)
    best_widthx = 0
    best_widthy = 0
    best_lamb = -Inf

    XX = sum(X .* X; dims=2)
    VV = sum(V .* V; dims=2)

    base_K = (XX .+ VV') .- (2 .* (X * V'))

    YY = sum(Y .* Y; dims=2)
    WW = sum(W .* W; dims=2)

    base_L = (YY .+ WW') .- (2 .* (Y * W'))

    # println("list_gwidthx is $list_gwidthx")
    # println("list_gwidthy is $list_gwidthy")

    for width_x in list_gwidthx
        K = exp.(-base_K ./ width_x)
        mean_k = mean(K, dims=1)
        Kt = K .- mean_k
        for width_y in list_gwidthy
            # println("width_x is $width_x")
            # println("width_y is $width_y")
            L = exp.(-base_L ./ width_y)
            try
                # mean
                mean_l = mean(L, dims=1)

                # biased
                u = mean(K .* L, dims=1) .- mean_k .* mean_l
                # cov
                Lt = L .- mean_l

                Snd_mo = Kt .* Lt
                Sig = (Snd_mo' * Snd_mo ./ n) .- (u' * u)

                lamb = nfsic_from_u_sig(u, Sig, n)
                println("lamb is $lamb")
                # println("best_lamb is $best_lamb")
                if lamb <= 0
                    error("The NSIC value was less than 0!")
                end
                if typeof(lamb) <: Complex
                    println("Lambda is complex. Truncating the imaginary part: $lamb")
                    lamb = real(lamb)
                end
                if lamb > best_lamb
                    best_lamb = lamb
                    best_widthx = width_x
                    best_widthy = width_y
                end
            catch _ continue end
        end
    end
    return best_widthx, best_widthy
end

function generic_optimize_locs_widths(X::AbstractArray, Y::AbstractArray,
        V0::AbstractArray, W0::AbstractArray, gwidthx0::Float64,
        gwidthy0::Float64, func_obj::Function;
        max_iter::Int64=400, V_step::Float64=1.0, W_step::Float64=1.0,
        gwidthx_step::Float64=1.0, gwidthy_step::Float64=1.0,
        batch_proportion::Float64=1.0, tol_fun::Float64=1e-3, 
        step_pow::Float64=0.5, reg::Float64=1e-5,
        gwidthx_lb::Float64=1e-3, gwidthx_ub::Float64=1e6,
        gwidthy_lb::Float64=1e-3, gwidthy_ub::Float64=1e6)

    """
    https://github.com/wittawatj/fsic-test/blob/master/fsic/indtest.py#L553-L723
    """

    if size(V0, 1) != size(W0, 1) 
        error("V0 and W0 must have the same number of rows J.")
    end

    constrain(var::Float64, lb::Float64, ub::Float64) = max(min(var, ub), lb)

    it = 1.0

    # heuristic to prevent step sizes from being too large
    max_gwidthx_step = minimum(std(X, dims=1)) / 2.0
    max_gwidthy_step = minimum(std(Y, dims=1)) / 2.0
    old_S = 0
    S = 0
    Vth = V0
    Wth = W0
    gwidthx_th = sqrt(gwidthx0)
    gwidthy_th = sqrt(gwidthy0)

    n = size(Y, 1)
    J = size(V0, 1)

    for t=1:max_iter
        # stochastic gradient ascent
        ind = sample(1:n, min(floor(Int64, batch_proportion * n), n);
            ordered=true, replace=false)
        try
            # Represent this as a function so I can get its gradient for later
            s(nt::NamedTuple) = func_obj(X[ind, :], Y[ind, :], nt.V, nt.W, nt.gwidthx,
                nt.gwidthy, reg, n, J)
            # @time (
            # println("The type of Vth is $(typeof(Vth))")
            # println("The type of Wth is $(typeof(Wth))")
            params = (V = Vth, W = Wth,
                gwidthx = gwidthx_th^2, gwidthy = gwidthy_th^2)
            S, gradient = Flux.withgradient(s, params)
            # println("calculated gradient!")
            g = gradient[1]

            g_V = g.V;             g_W = g.W
            g_gwidthx = g.gwidthx; g_gwidthy = g.gwidthy

            # updates
            Vth .+= (V_step / it^step_pow / sqrt(mapreduce(x -> x^2, +, g_V))) .* g_V
            Wth .+= (W_step / it^step_pow / sqrt(mapreduce(x -> x^2, +, g_W))) .* g_W
            it += 1
            gwidthx_th = constrain(
                gwidthx_th + gwidthx_step * sign(g_gwidthx) * 
                    min(abs(g_gwidthx), max_gwidthx_step) / it^step_pow,
                    sqrt(gwidthx_lb), sqrt(gwidthx_ub)
            )
            gwidthy_th = constrain(
                gwidthy_th + gwidthy_step * sign(g_gwidthy) * 
                    min(abs(g_gwidthy), max_gwidthy_step) / it^step_pow,
                    sqrt(gwidthy_lb), sqrt(gwidthy_ub)
            )

            if t >= 4 && abs(old_S - S) <= tol_fun break end
            old_S = S
            # )
        catch e
            println("Exception occurred during gradient descent. Stop optimization.")
            println("Return the value from previous iter. ")
            throw(e)
            break
        end

        if t >= 0  return (Vth, Wth, gwidthx_th, gwidthy_th)
        else       return (V0, W0, gwidthx0, gwidthy0) # Probably an error occurred in the first iteration.
        end
    end
end

function func_obj(Xth::AbstractArray, Yth::AbstractArray, Vth::AbstractArray,
        Wth::AbstractArray, gwidthx_th::Float64, gwidthy_th::Float64,
        regth::Float64, n::Int64, J::Int64)
    """
    https://github.com/wittawatj/fsic-test/blob/master/fsic/indtest.py#L242-L277
    """
    # println("Begin function func_obj()")
    # println("Calculating diag_regth...")
    diag_regth = regth .* CuMatrix(1.0I, J, J)
    # println("Calculating Kth...")
    Kth = rbf_dot(Xth, Vth, gwidthx_th)
    # println("Calculating Lth...")
    Lth = rbf_dot(Yth, Wth, gwidthy_th)

    # println("Calculating mean_k...")
    mean_k = mean(Kth, dims=1)
    # println("Calculating mean_l...")
    mean_l = mean(Lth, dims=1)
    # println("Calculating KLth...")
    KLth = Kth .* Lth
    # println("Calculating u...")
    u = mean(KLth, dims=1) .- mean_k .* mean_l

    # println("Calculating Kth_norm...")
    Kth_norm = Kth .- mean_k
    # println("Calculating Lth_norm...")
    Lth_norm = Lth .- mean_l
    # Gam is n x J
    # println("Calculating Gam...")
    Gam = (Kth_norm .* Lth_norm .- u) .- mean(Kth_norm .* Lth_norm .- u, dims=1)
    # println("Calculating Sig...")
    Sig = Gam' * Gam ./ Float64(n)
    # println("Type of Sig is: $(typeof(Sig))")
    # println("Calculating output...")
    out = dot(gpu(inv(cpu(Sig .+ diag_regth))) * u', u) / Float64(n)
    # println("Done!")
    out
end

# function func_obj_with_grad(Xth::AbstractArray, Yth::AbstractArray, Vth::AbstractArray,
#         Wth::AbstractArray, gwidthx_th::Float64, gwidthy_th::Float64,
#         regth::Float64, n::Int64, J::Int64)
#     """
#     https://github.com/wittawatj/fsic-test/blob/master/fsic/indtest.py#L242-L277
#     """
#     diag_regth = regth .* eye(J)
#     # println("Type of diag_regth is: $(typeof(diag_regth))")
#     Kth = rbf_dot(Xth, Vth, gwidthx_th)
#     Lth = rbf_dot(Yth, Wth, gwidthy_th)

#     mean_k = mean(Kth, dims=1)
#     mean_l = mean(Lth, dims=1)
#     KLth = Kth .* Lth
#     u = mean(KLth, dims=1) .- mean_k .* mean_l

#     Kth .-= mean_k
#     Lth .-= mean_l
#     # Gam is n x J
#     Gam = Kth .* Lth .- u
#     mean_gam = mean(Gam, dims=1)
#     Gam .-= mean_gam
#     Sig = Gam' * Gam ./ Float64(n)
#     # println("Type of Sig is: $(typeof(Sig))")
#     cuinv(Sig .+ diag_regth) * u' * u / Float64(n)
# end

function cuinv(A::CuMatrix{Float64})
    """
    https://discourse.julialang.org/t/cuda-matrix-inverse/53341
    """
    if size(A, 1) != size(A, 2) throw(ArgumentError("Matrix not square.")) end
    println("Calculating B...")
    B = eye(size(A, 1))
    # println("type of B is: $(typeof(B))")
    # B = CuArray(Matrix{T}(I(size(m,1))))
    println("Calculating A_rf and ipiv...")
    A_rf, ipiv = CUDA.CUSOLVER.getrf!(A)
    println("Calculating output...")
    out = CUDA.CUSOLVER.getrs!('N', A_rf, ipiv, Float64.(B))
    println("done!")
    return out
end

# ChainRulesCore.rrule(::typeof(reduce), ::typeof(+), x::AbstractArray; dims=:) = ChainRulesCore.rrule(sum, x; dims)