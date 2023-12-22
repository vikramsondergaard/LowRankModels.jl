import Distributions: Gamma

export evaluate, grad, get_independence_criterion, HSIC, NFSIC

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

function get_independence_criterion(Y::AbstractArray{Float32, 2}, ic::DataType)
    if ic == HSIC
        return get_hsic(Y)
    elseif ic == NFSIC
        return get_nfsic(Y)
    else
        error("Independence criterion $ic not implemented yet!")
    end
end
get_independence_criterion(Y::AbstractArray{Float64, 2}, ic::DataType) =
    get_independence_criterion(Float32.(Y), ic)
get_independence_criterion(Y::AbstractArray{Float64, 1}, ic::DataType) =
    get_independence_criterion(Float32.(Y)[:, :], ic)

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
    G = mapreduce(identity, +, pattern1 .* pattern1; dims=2)
    H = mapreduce(identity, +, pattern2 .* pattern2; dims=2)

    # Project both matrices into a common space
    H = (G .+ H') .- (2 .* (pattern1 * pattern2'))

    # Apply Gaussian distance calculation
    map(exp, -H ./ (2 * deg^2))
end

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
end
function get_nfsic(Y::AbstractArray)
    # width_y = get_width(Y)
    idxs = rand(1:size(Y, 1), 1250)
    W = CuArray(Y[idxs, :])
    get_nfsic(Y, W, Float64(var(Y)))
end
function get_nfsic(Y::AbstractArray, W::AbstractArray)
    get_nfsic(Y, W, Float64(var(Y)))
end
function get_nfsic(Y::AbstractArray, W::AbstractArray, width::Float64)
    L = rbf_dot(Y, W, width)
    mean_l = mean(L; dims=1)
    NFSIC(L, mean_l, L .- mean_l)
end
# get_nfsic(Y::AbstractArray{Float64}, W::AbstractArray{Float32}) = get_nfsic(Float32.(Y), W)
# get_nfsic(Y::AbstractArray{Float32}, W::AbstractArray{Float64}) = get_nfsic(Y, Float32.(W))
# get_nfsic(Y::AbstractArray{Float64}, W::AbstractArray{Float64}) = get_nfsic(Float32.(Y), Float32.(W))

function evaluate(nfsic::NFSIC, X::AbstractArray{Float32}; optim=false, reg=1)
    n = size(X, 1)
    idxs = rand(1:n, 1250)
    V = CuArray(X[idxs, :])
    evaluate(nfsic, X, V)
end

function evaluate(nfsic::NFSIC, X::AbstractArray{Float32},
        V::AbstractArray{Float32}; 
        width_x::Float64=Float64(var(X)), optim=false, reg=1)
    n = size(X, 1)
    J = size(V, 1)
    # width_x = get_width(X)
    
    K = rbf_dot(X, V, width_x) # n x J
    L = nfsic.L                        # n x J
    
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
    nfsic_from_u_sig(Float32.(u), Float32.(Sig), n, reg)
end

function nfsic_from_u_sig(u::AbstractArray{Float32, 2}, Sig::AbstractArray{Float32, 2}, n::Int64, reg::Int=0)
    J = length(u)
    if J == 1
        r = max(reg, 1250)
        s = Float64(n) .* dot(u, ones(1)).^2 / (r .+ dot(Sig, ones(1)))
    else
        if reg <= 0
            try
                s = n * dot(Sig \ u, u)
            catch _
                try
                    # singular matrix
                    # eigen decompose
                    evals, eV = eig(Sig)
                    evals = max.(0, evals)
                    # find the non-zero second smallest eigenvalue
                    snd_small = minimum(filter(e -> e > 0, evals))
                    evals = max.(snd_small, evals)

                    # reconstruct Sig
                    Sig = dot.(eV, diag(evals)) * eV'
                    # try again
                    s = n * dot(Sig \ u, u)
                catch _
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

function grad(nfsic::NFSIC, X::AbstractArray{Float32}; optim=false, reg=1)
    n = size(X, 1)

    if !optim
        idxs = rand(1:n, 1250)
        V = CuArray(X[idxs, :])
    end

    J = size(V, 1)
    width_x = Float64(var(X))
    
    K = rbf_dot(X, V, width_x)              # n x J
    Km = -1 .* K .* (X .- V') ./ width_x^2  # gradient
    L = nfsic.L                             # n x J
    
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
        n_test_locs::Int=5, max_iter::Int=400, V_step::Int=1, W_step::Int=1,
        gwidthx_step::Int=1, gwidthy_step::Int=1, batch_proportion::Float64=1,
        tol_fun::Float64=1.0e-3, step_pow::Float64=0.5, seed::Int=1,
        reg::Float64=1.0e-5, gwidthx_lb::MaybeFloat64=nothing,
        gwidthy_lb::MaybeFloat64=nothing, gwidthx_ub::MaybeFloat64=nothing,
        gwidthy_ub::MaybeFloat64=nothing)
    J = n_test_locs
    # Use grid search to initialise the gwidths for both X, Y
    n_gwidth_cand = 5
    gwidth_factors = CuArray(LinRange(-3, 3, n_gwidth_cand)).^2
    medx2 = meddistance(X, 1000)^2
    medy2 = meddistance(Y, 1000)^2

    V, W = init_check_subset(X, Y, medx2 * 2, medy2 * 2, J)
    best_widthx, best_widthy = nfsic_grid_search_kernel(X, Y, V, W,
        medx2 * gwidth_factors, medy2 * gwidth_factors)
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
    n = size(X, 1)

    # from the joint
    best_obj_joint = 0
    best_V_joint::Union{AbstractArray, Nothing} = nothing
    best_W_joint::Union{AbstractArray, Nothing} = nothing
    for _=1:n_cand
        V, W = init_locs_joint_subset(X, Y, n_test_locs)
        if subsample < n
            I = sample(1:n, n_test_locs)
            nfsic = get_nfsic(Y[I, :], W, widthy)
            obj_joint = evaluate(nfsic, X[I, :], V; width_x=widthx, reg=0)
        else
            nfsic = get_nfsic(Y, W, widthy)
            obj_joint = evaluate(nfsic, X, V; width_x=widthx, reg=0)
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
            I = sample(1:n, n_test_locs)
            nfsic = get_nfsic(Y[I, :], W, widthy)
            obj_prod = evaluate(nfsic, X[I, :], V; width_x=widthx, reg=0)
        else
            nfsic = get_nfsic(Y, W, widthy)
            obj_prod = evaluate(nfsic, X, V; width_x=widthx, reg=0)
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

    XX = mapreduce(identity, +, X .* X; dims=2)
    VV = mapreduce(identity, +, V .* V; dims=2)

    base_K = (XX .+ VV') .- (2 .* (X * V'))

    YY = mapreduce(identity, +, Y .* Y; dims=2)
    WW = mapreduce(identity, +, W .* W; dims=2)

    base_L = (YY .+ WW') .- (2 .* (Y * W'))

    for width_x in list_gwidthx
        K = map(exp, -base_K ./ (2 * width_x))
        Kt = K .- mean(K, dims=1)
        for width_y in list_gwidthy
            L = map(exp, -base_L ./ (2 * width_y))
            try
                # mean
                mean_l = mean(L, dims=1)

                # biased
                u = mean(K .* L, dims=1) - mean_k .* mean_l
                # cov
                Lt = L .- mean_l

                Snd_mo = Kt .* Lt
                Sig = (Snd_mo' * Snd_mo ./ n) - (u' * u)

                lamb = nfsic_from_u_sig(u, Sig, n)
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
            catch _ continue
            end
        end
    end
    return best_widthx, best_widthy
end

function generic_optimize_locs_widths(X::AbstractArray, Y::AbstractArray,
        V0::AbstractArray, W0::AbstractArray, gwidthx0::Float32,
        gwidthy0::Float32, func_obj::Float32;
        max_iter::Int64=400, V_step::Float64=1.0, W_step::Float64=1.0,
        gwidthx_step::Float64=1.0, gwidthy_step::Float64=1.0,
        batch_proportion::Float64=1.0, tol_fun::Float64=1e-3, 
        step_pow::Float64=0.5, reg::Float64=1e-5, seed::Int64=101,
        gwidthx_lb::Float64=1e-3, gwidthx_ub::Float64=1e6,
        gwidthy_lb::Float64=1e-3, gwidthy_ub::Float64=1e6)
    if size(V0, 1) != size(W0, 1) 
        error("V0 and W0 must have the same number of rows J.")
    end

    it = 1
    s(nt::NamedTuple) = func_obj(nt.Xth, nt.Yth, nt.Vth, nt.Wth, nt.gwidthx_th,
        nt.gwidthy_th, nt.regth, nt.n, nt.J)
    params = (Xth = X, Yth = Y, Vth = V0, Wth = W0, gwidthx_th = gwidthx0,
        gwidthy_th = gwidthy0, regth = reg, n = size(X, 1), J = size(V0, 1))
    g = gradient(s, params)[1]

    model = Chain(
        
    )
end

function func_obj(Xth::AbstractArray, Yth::AbstractArray, Vth::AbstractArray,
        Wth::AbstractArray, gwidthx_th::Float32, gwidthy_th::Float32,
        regth::Float32, n::Float32, J::Int32)
    """
    https://github.com/wittawatj/fsic-test/blob/master/fsic/indtest.py#L242-L277
    """
    diag_regth = regth .* eye(J) 
    Kth = rbf_dot(Xth, Vth, gwidthx_th)
    Lth = rbf_dot(Yth, Wth, gwidthy_th)

    mean_k = mean(Kth, dims=1)
    mean_l = mean(Lth, dims=1)
    KLth = Kth .* Lth
    u = mean(KLth, dims=1) .- mean_k .* mean_l

    Kth .-= mean_k
    Lth .-= mean_l
    # Gam is n x J
    Gam = Kth .* Lth .- u
    mean_gam = mean(Gam, dims=1)
    Gam .-= mean_gam
    Sig = Gam' * Gam ./ n
    inv(Sig .+ diag_regth) * u' * u / n
end