import Distributions: Gamma
import LinearAlgebra: dot, tril, I, trace, Diagonal
import Statistics: median

function rbf_dot(pattern1::AbstractArray, pattern2::AbstractArray, deg::Float64)
    # Get the size of both matrices for the RBF
    size1 = size(pattern1, 1)
    size2 = size(pattern2, 1)

    G = [dot(pattern1[i, :], pattern1[i, :]) for i=1:size1]
    H = [dot(pattern2[i, :], pattern2[i, :]) for i=1:size2]

    Q = repeat(G, outer=[1, size2])
    R = repeat(H', outer=[size1, 1])

    H = Q .+ R .- (2 * dot(pattern1, pattern2'))

    H = exp(-H ./ (2 * deg^2))

    H
end

function get_width(M::AbstractArray)
    n = size(M, 1)

    G = [dot(M[i, :], M[i, :]) for i=1:n]
    Q = repeat(G, outer=[1, n])
    R = repeat(G', outer=[n, 1])

    dists = Q .+ R .- (2 .* (M * M'))
    dists = dists .- tril(dists)
    dists = reshape(dists, (n^2, 1))

    sqrt(0.5 * median(filter(d -> d > 0, dists)))
end

function hsic_gam(X, Y, alph=0.5)
    n = size(X, 1)
    width_x = get_width(X)
    width_y = get_width(Y)

    bone = ones(Float64, n, 1)
    
    H = Matrix(I, n, n) - ones(Float64, n, n) ./ n

    K = rbf_dot(X, X, width_x)
    L = rbf_dot(Y, Y, width_y)

    Kc = (H * K) * H
    Lc = (H * L) * H

    test_stat = sum(Kc' .* Lc) / n

    var_HSIC = (Kc .* Lc ./ 6) .^ 2
    var_HSIC = (sum(var_HSIC) - trace(var_HSIC)) / n / (n - 1)
    var_HSIC = var_HSIC * 72 * (n - 4) * (n - 5) / n / (n - 1) / (n - 2) / (n - 3)

    K = K .- Diagonal(K)
    L = L .- Diagonal(L)

    mu_x = ((bone' * K) * bone) / n / (n - 1)
    mu_y = ((bone' * L) * bone) / n / (n - 1)

    m_HSIC = (1 + mu_x * mu_y - mu_x - mu_y) / n

    al = m_HSIC^2 / var_HSIC
    bet = var_HSIC * n / m_HSIC

    thresh = quantile(Gamma(al, bet), 1 - alph)

    test_stat, thresh
end