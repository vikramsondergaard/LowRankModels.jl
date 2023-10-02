using LowRankModels
import Random

Random.seed!(1)

losses₂ = [HingeLoss(), QuadLoss()]
losses₃ = [OvALoss(3, bin_loss=HingeLoss()), HingeLoss(),
           QuadLoss(), OrdinalHingeLoss(4)]
losses₄ = [OvALoss(4, bin_loss=HingeLoss()), HingeLoss(),
           OvALoss(6, bin_loss=HingeLoss()), QuadLoss(),
           QuadLoss(), OrdinalHingeLoss(3), OrdinalHingeLoss(5)]

A₂_bool = [true, false, true]
A₂_real = [-0.82745, 1.26234, -0.29391]

A₃  = [1 false  0.49515 1
       1 false -0.15264 2
       1 true   0.77662 1
       2 true  -1.65449 4
       2 false -0.17560 4
       3 false  0.90025 2]

A₃_cat = [1, 1, 1, 2, 2, 3]
A₃_bool = [-1, -1, 1, 1, -1, -1]
A₃_real = [0.49515, -0.15264, 0.77662, -1.65449, -0.17560, 0.90025]
A₃_ord = [1, 2, 1, 4, 4, 2]

A₄  = [1 false 6 -0.83625 -0.62522 2 3
       1 false 4  1.62911  0.62544 3 3
       1 true  5  1.19634 -0.67659 2 4
       1 true  2 -0.19417  0.05815 2 3
       2 false 3 -0.54548 -0.25093 2 3
       2 false 2  0.20709 -0.73999 1 1
       2 true  1  1.05964 -1.04975 1 2
       3 false 5 -0.41204  0.05053 2 4
       3 false 1  0.80596  0.56225 2 5
       4 false 3  0.28682 -0.57457 1 1]

s = 1
p = GradParams(1, max_iter=200, abs_tol=0.0000001, min_stepsize=0.001)

function test(Aexp, Aout)
    m,n = size(Aexp)
    mout,nout = size(Aout)
    @assert m == mout && n == nout
    for i=1:m
        for j=1:n
            println("At row $i, column $j")
            println("The value of the GLRM is: $(Aexp[i, j])")
            println("The value of the fair GLRM is: $(Aout[i, j])")
            @assert isapprox(Aexp[i, j], Aout[i, j], atol=0.001)
        end
    end
end

function test_small()
    A₂ = Any[A₂_bool A₂_real]
    m,n = size(A₂)
    k = 1
    X_init = randn(k, m)
    Y_init = randn(k, n)

    groups = partition_groups(A₂, s, 2)
    weights = [Float64(length(g)) / m for g in groups]
    # weights = [1 / length(groups) for _ in groups]

    glrm = GLRM(A₂, losses₂, ZeroReg(), ZeroReg(), k, X=deepcopy(X_init), Y=deepcopy(Y_init))
    glrmX, glrmY, _ = fit!(glrm, params=p, verbose=true)
    println("successfully fit vanilla GLRM")

    fglrm = FairGLRM(A₂, losses₂, ZeroReg(), ZeroReg(), k, s,
    WeightedLogSumExponentialLoss(10^(-6), weights),
    X=deepcopy(X_init), Y=deepcopy(Y_init), Z=groups)
    fglrmX, fglrmY, _ = fit!(fglrm, params=p, verbose=true)
    println("successfully fit fair GLRM")

    Aexp = glrmX' * glrmY
    Aout = fglrmX' * fglrmY
    test(Aexp, Aout)
    println("Passed test_small()!")
end

test_small()