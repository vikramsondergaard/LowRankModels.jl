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

mutable struct HSIC
    hsic::Float64
    HLH::CuArray{Float64, 2}
    K::CuArray{Float64, 2}
    X::CuArray{Float64, 2}
end
HSIC(Y::AbstractArray{Float64, 2}) = begin
    n, dim_y = size(Y)
    H = CuMatrix(I, n, n) .- CUDA.ones(Float64, n, n) ./ n
    width_y = get_width(Y)
    L = rbf_dot(Y, Y, width_y)
    HLH = H * L * H
    HSIC(0.0, HLH, CUDA.zeros(n, n), CUDA.zeros(n, 1))
end

function rbf_dot(pattern1::AbstractArray, pattern2::AbstractArray, deg::Float64)
    # Get the size of both matrices for the RBF
    size1 = size(pattern1, 1)
    size2 = size(pattern2, 1)

    # Get sum of squares along the rows
    G = mapreduce(identity, +, pattern1 .* pattern1; dims=2)
    H = mapreduce(identity, +, pattern2 .* pattern2; dims=2)

    # 
    Q = repeat(G, outer=[1, size2])
    R = repeat(H', outer=[size1, 1])

    H = Q .+ R .- (2 .* (pattern1 * pattern2'))

    H = map(exp, -H ./ (2 * deg^2))

    H
end

function get_width(M::AbstractArray)
    n = size(M, 1)

    G = mapreduce(identity, +, M .* M; dims=2)
    Q = repeat(G, outer=[1, n])
    R = repeat(G', outer=[n, 1])

    dists = Q .+ R .- (2 .* (M * M'))
    dists .-= tril(dists)
    dists = reshape(dists, (n^2, 1))
    filt_dists = filter(d -> d > 0, dists)

    if isempty(filt_dists) 
        return 0 
    else 
        return sqrt(0.5 * median(filt_dists)) 
    end
end

function hsic_gam!(hsic::HSIC, X::AbstractArray;
        alph::Float64=0.5)
    n = size(X, 1)
    if n == 1 return 0 end
    width_x = get_width(X)
    if width_x == 0 return 0 end

    K = rbf_dot(X, X, width_x)
    hsic.K = K
    hsic_mat = K * hsic.HLH

    test_stat = reduce(+, diag(hsic_mat))

    hsic.hsic = test_stat

    test_stat / n^2
end

function hsic_gam!(hsic::HSIC, X::AbstractArray, e::Int)
    # Get σ to get the width of the newly changed distribution
    n = size(X, 1)
    width_x = get_width(X)
    # The new RBF only differs from the old RBF for the changed value e: so we
    # only need a vector, not a matrix
    K = rbf_dot(X, X, width_x)
    # Old RBF needs to be taken for calculating difference
    old_rbf = hsic.K[e, :]
    new_rbf = K[e, :]
    rbf_diff = new_rbf - old_rbf
    a = CuArray([i == e for i=1:n])
    A = broadcast(|, a, a')
    HLH = hsic.HLH .* A
    # These parts of the trace are the only differences from the old HSIC
    # Note: this has a double-up at (e, e)
    trace = reduce(+, broadcast(*, HLH, rbf_diff)) / n^2
    # Update the trace value so I can get it back later
    hsic.hsic += trace
    hsic.X = X
    hsic.K = K
    hsic.hsic
end

function hsic_grad!(hsic::HSIC, X::AbstractArray)
    n = size(X, 1)
    if n == 1 return zeros(n) end
    dim_x = size(X, 2)
    width_x = get_width(X)
    if width_x == 0 return zeros(n) end

    K = rbf_dot(X, X, width_x)

    M = broadcast(-, X, X')

    Km = K .* M # Hadamard product
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