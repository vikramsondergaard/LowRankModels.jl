using LowRankModels, Statistics

A₂_bool = [1, -1, 1]
A₂_real = [-0.82745, 1.26234, -0.29391]

A₃_cat = [1, 1, 1, 2, 2, 3]
A₃_bool = [false, false, true, true, false, false]
A₃_real = [0.49515, -0.15264, 0.77662, -1.65449, -0.17560, 0.90025]
A₃_ord = [1, 2, 1, 4, 4, 2]

normalise(A::AbstractArray) = A .- mean(A)

function normalise!(A::Matrix{Any})
    mu = mean(A, dims=1)
    broadcast!(-, A, A, mu)
end

function test_prox(Aexp, X, a)
    orth_reg = OrthogonalReg(a)
    Aout = prox(orth_reg, X)
    if length(size(Aexp)) == 1
        @assert length(Aexp) == length(Aout)
        for i=1:length(Aexp)
            err_str = "At index $i\nExpected: $(Aexp[i])\nGot: $(Aout[i])"
            @assert isapprox(Aexp[i], Aout[i], atol=0.001) err_str
        end
    else
        m,n = size(Aexp)
        mout,nout = size(Aout)
        @assert m == mout && n == nout
        for i=1:m
            for j=1:n
                err_str = "At row $i, column $j\nExpected: $(Aexp[i, j])\nGot: $(Aout[i, j])"
                @assert isapprox(Aexp[i, j], Aout[i, j], atol=0.001) err_str
            end
        end
    end 
end

function test_orthogonal(X, exp, a)
    orth_reg = OrthogonalReg(a)
    val = evaluate(orth_reg, X)
    if exp == Inf
        @assert val == Inf "Expected $X and protected characteristic $a to be non-orthogonal, but got that they were orthogonal!"
    else
        @assert val == 0 "Expected $X and protected characteristic $a to non-orthogonal, but got that they were non-orthogonal!"
    end
end

function test_small()
    A₂ = Any[A₂_bool A₂_real]
    normalise!(A₂)
    X = [3, 3, 3]
    test_orthogonal(X, 0, A₂[:, 1])
    X = [3, 3, 2]
    test_orthogonal(X, Inf, A₂[:, 1])
    X = [1.0 / 7.0, -5.0 / 7.0, 4.0 / 7.0]
    test_prox([-3.0 / 14.0, 0, 3.0 / 14.0], convert(Array, X), A₂[:, 1])
    println("Passed test_small()!")
end

function test_fit()
    A₃ = Any[A₃_cat A₃_bool A₃_real A₃_ord]
    m = size(A₃, 1)
    # normalise!(A₃)
    println("Normalised A₃ is")
    display(A₃)
    prot_cat = 2
    losses₃ = [OvALoss(3, bin_loss=HingeLoss()), HingeLoss(),
        QuadLoss(), OrdinalHingeLoss(4)]
    groups = partition_groups(A₃, prot_cat, 2)
    p = Params(1, max_iter=200, abs_tol=0.0000001, min_stepsize=0.001)
    display(A₃[:, prot_cat])
    fglrm = FairGLRM(A₃, losses₃, OrthogonalReg(normalise(A₃[:, prot_cat])), ZeroReg(), 2, prot_cat,
        WeightedLogSumExponentialLoss(10^(-6), [Float64(length(groups[i])) / m for i=1:2]),
        Z=groups)
    # fglrm = FairGLRM(A₃, losses₃, ZeroReg(), ZeroReg(), 2, prot_cat,
    #     WeightedLogSumExponentialLoss(10^(-6), [Float64(length(groups[i])) / m for i=1:2]),
    #     Z=groups)
    fglrmX, fglrmY, _ = fit!(fglrm, params=p, verbose=true)
    println("fglrmX is:")
    display(fglrmX)
    println("fglrmY is:")
    display(fglrmY)
    println("Their product is")
    display(fglrmX' * fglrmY)
    println("Passed test_fit()!")
end

test_small()
test_fit()