import Distributions: Gamma

export hsic_gam, hsic_grad

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

function rbf_dot(pattern1::AbstractArray, pattern2::AbstractArray, deg::Float64)
    # Get the size of both matrices for the RBF
    size1 = size(pattern1, 1)
    size2 = size(pattern2, 1)

    # Get sum of squares along the rows
    G = [dot(pattern1[i, :], pattern1[i, :]) for i=1:size1]
    H = [dot(pattern2[i, :], pattern2[i, :]) for i=1:size2]

    # 
    Q = repeat(G, outer=[1, size2])
    R = repeat(H', outer=[size1, 1])

    H = Q .+ R .- (2 .* (pattern1 * pattern2'))

    H = map(exp, -H ./ (2 * deg^2))

    H
end

function get_width(M::AbstractArray)
    n = size(M, 1)

    G = [dot(M[i, :], M[i, :]) for i=1:n]
    Q = repeat(G, outer=[1, n])
    R = repeat(G', outer=[n, 1])

    dists = Q .+ R .- (2 .* (M * M'))
    dists .-= tril(dists)
    dists = reshape(dists, (n^2, 1))

    sqrt(0.5 * median(filter(d -> d > 0, dists)))
end

function hsic_gam(X::AbstractArray, Y::AbstractArray, alph::Float64=0.5)
    n = size(X, 1)
    if n == 1 return 0 end
    width_x = get_width(X)
    width_y = get_width(Y)

    bone = ones(Float64, n, 1)
    
    H = Matrix(I, n, n) - ones(Float64, n, n) ./ n

    K = rbf_dot(X, X, width_x)
    L = rbf_dot(Y, Y, width_y)

    Kc = (H * K) * H
    Lc = (H * L) * H

    test_stat = sum(Kc' .* Lc) / n

    # var_HSIC = (Kc .* Lc ./ 6) .^ 2
    # var_HSIC = (sum(var_HSIC) - tr(var_HSIC)) / n / (n - 1)
    # var_HSIC = var_HSIC * 72 * (n - 4) * (n - 5) / n / (n - 1) / (n - 2) / (n - 3)

    # K = K .- Diagonal(K)
    # L = L .- Diagonal(L)

    # mu_x = ((bone' * K) * bone) / n / (n - 1)
    # mu_y = ((bone' * L) * bone) / n / (n - 1)

    # m_HSIC = (1 .+ mu_x .* mu_y .- mu_x .- mu_y) ./ n

    # al = m_HSIC^2 / var_HSIC
    # bet = var_HSIC * n ./ m_HSIC

    # println("Size of al is $(size(al))")
    # println("al is $(al[1])")
    # println("Size of bet is $(size(bet))")
    # println("bet is $(bet[1])")

    # if size(al) == (1, 1) && size(bet) == (1, 1)
    #     thresh = quantile(Gamma(al[1], bet[1]), 1 - alph)
    # else
    #     thresh = quantile(Gamma(al, bet), 1 - alph)
    # end

    test_stat
end

function hsic_grad(X::AbstractArray, Y::AbstractArray)
    n = size(X, 1)
    if n == 1 return zeros(n) end
    dim_x = size(X, 2)
    width_x = get_width(X)
    width_y = get_width(Y)
    
    H = Matrix(I, n, n) - ones(Float64, n, n) ./ n

    K = rbf_dot(X, X, width_x)
    L = rbf_dot(Y, Y, width_y)

    M = zeros(n, n, dim_x)
    for i=1:n
        for j=1:n
            for q=1:dim_x
                M[i, j, q] = X[i, q] - X[j, q]
            end
        end
    end

    Lc = (H * L) * H

    G = zeros(n, dim_x)
    for i=1:n
        for q=1:dim_x
            Km = K .* M[:, :, q]
            Kc = (H * Km) * H
            test_stat = 2 * sum(Kc' .* Lc) / n / width_x^2
            G[i, q] = test_stat
        end
    end

    G
end