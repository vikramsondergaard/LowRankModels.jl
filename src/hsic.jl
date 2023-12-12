import Distributions: Gamma

export hsic_gam, hsic_grad, HSIC

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
    # Get sum of squares along the rows
    G = mapreduce(identity, +, pattern1 .* pattern1; dims=2)
    H = mapreduce(identity, +, pattern2 .* pattern2; dims=2)

    H = (G .+ H') .- (2 .* (pattern1 * pattern2'))

    H = map(exp, -H ./ (2 * deg^2))

    H
end

function rbf_dot(X::AbstractArray, deg²::Float64)
    G = mapreduce(identity, +, X .* X; dims=2)
    H = (G .+ G') .- (2 .* (X * X'))
    map(exp, -H ./ (2 * deg²))
end

function get_width(M::AbstractArray)
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

function hsic_gam!(hsic::HSIC, X::AbstractArray;
        alph::Float64=0.5)
    n = size(X, 1)
    # if n == 1 return 0 end
    # width_x = get_width(X)
    # if width_x == 0 return 0 end

    K = rbf_dot(X, var(X))
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

function hsic_grad!(hsic::HSIC, X::AbstractArray)
    n = size(X, 1)
    if n == 1 return zeros(n) end
    dim_x = size(X, 2)
    width_x = var(X)
    if width_x == 0 return zeros(n) end

    Km = rbf_dot(X, X, width_x) .* broadcast(-, X, X') # Hadamard product
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
    L::AbstractArray{Float32, 2}
end
function get_nfsic(Y::AbstractArray{Float32}, W::AbstractArray{Float32})
    width_y = get_width(Y)
    L = rbf_dot(Y, W, width_y)
    NFSIC(L)
end
get_nfsic(Y::AbstractArray{Float64}, W::AbstractArray{Float32}) = get_nfsic(Float32.(Y), W)
get_nfsic(Y::AbstractArray{Float32}, W::AbstractArray{Float64}) = get_nfsic(Y, Float32.(W))
get_nfsic(Y::AbstractArray{Float64}, W::AbstractArray{Float64}) = get_nfsic(Float32.(Y), Float32.(W))

function calc_nfsic(nfsic::NFSIC, X::AbstractArray{Float32}, V::AbstractArray{Float32}; reg=0)
    n = size(X, 1)
    J = size(V, 1)
    width_x = get_width(X)
    
    K = rbf_dot(X, V, width_x) # n x J
    L = nfsic.L                # n x J
    
    # mean
    mean_k = mean(K; dims=1)
    mean_l = mean(L; dims=1)

    # biased
    u = CuArray(mean(K .* L; dims=1) - (mean_k .* mean_l))
    # cov
    # Generic covariance
    Kt = K .- mean_k
    Lt = L .- mean_l
    
    Snd_mo = Kt .* Lt
    Sig = (Snd_mo' * Snd_mo ./ n) - (u' * u)
    nfsic_from_u_sig(Float32.(u), Float32.(Sig), n, reg)
end

function nfsic_from_u_sig(u::AbstractArray{Float32, 2}, Sig::AbstractArray{Float32, 2}, n::Int64, reg::Int=0)
    J = length(u)
    if J == 1
        r = max(reg, 0)
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