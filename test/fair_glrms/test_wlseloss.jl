using LowRankModels

losses₂ = [OvALoss(2, bin_loss=HingeLoss()), QuadLoss()]
losses₃ = [OvALoss(3, bin_loss=HingeLoss()), HingeLoss(),
           QuadLoss(), OrdinalHingeLoss(4)]
losses₄ = [OvALoss(4, bin_loss=HingeLoss()), HingeLoss(),
           OvALoss(6, bin_loss=HingeLoss()), QuadLoss(),
           QuadLoss(), OrdinalHingeLoss(3), OrdinalHingeLoss(5)]

A₂  = [1 -0.82745
       2  1.26234
       1 -0.29391]

XY₂ = [2 -0.90945
       2  1.26234
       1 -1.58446]

Z₂  = [1.5  0.00672/2
       1    0.0
       0.5  1.66552/2]

A₃  = [1 false  0.49515 1
       1 false -0.15264 2
       1 true   0.77662 1
       2 true  -1.65449 4
       2 false -0.17560 4
       3 false  0.90025 2]

XY₃ = [1 -1     0.49515 4
       1 -1    -0.15264 4
       2 -1     0.77662 2
       2 -1    -0.77418 1
       1 -1    -0.17560 4
       3 -1     0.04921 3]

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

XY₄ = [1  1    3 -0.83625 -0.62522 1 1
       3  1    4  0.29176  0.62544 2 3
       1  1    2  1.19634  1.51360 2 3
       4  1    5 -0.19417 -0.93572 2 3
       1  1    4 -1.20147 -1.27625 2 2
       2 -1    6 -1.57535 -1.56085 1 5
       2  1    5 -0.77495 -1.04975 1 4
       4 -1    5 -0.41204 -0.08579 2 3
       4 -1    2  0.80596 -0.08575 1 2
       4  1    4  0.28682  1.21956 3 1]

function test(l::WeightedLogSumExponentialLoss, losses::Array{<:Loss, 1}, XY,
        A, Z, observed_features, wlseexp::Float64;
        yidxs = get_yidxs(losses))
    wlseout = evaluate(l, losses, XY, A, Z, observed_features, yidxs=yidxs)
    println("Expected: $wlseexp")
    println("Got: $wlseout")
    @assert isapprox(wlseexp, wlseout, atol=0.0001)
end

function test_alpha()
    αs = [10^(-6), 10^(-5), 3 * 10^(-3), 10^(-2), 10^(-1), 5 * 10^(-1), 1, 5,
          10, 20, 60, 10^2, 10^3, 10^4, 10^5]
end

function test_small()
    wlse = WeightedLogSumExponentialLoss(10^(-6), [0.5, 0.5])
    Z = [[1, 3], [2]]
    wlseexp = 1.918060421540249
    test(wlse, losses₂, XY₂, A₂, Z, [1:2 for i=1:3], wlseexp, yidxs=1:2)
    println("Completed test_small()!")
end

function test_medium()
    wlse = WeightedLogSumExponentialLoss(10^(-6), [1.0/2.0, 1.0/3.0, 1.0/6.0])
    Z = [1:3, 4:5, [6]]
    wlseexp = 6.416537443555668
    test(wlse, losses₃, XY₃, A₃, Z, [1:4 for i=1:6], wlseexp, yidxs=1:4)
    println("Completed test_medium()!")
end

function test_large()
    weights = [2.0/5.0, 3.0/10.0, 1.0/5.0, 1.0/10.0]
    wlse = WeightedLogSumExponentialLoss(10^(-6), weights)
    Z = [1:4, 5:7, 8:9, [10]]
    wlseexp = 16.492883691616235
    test(wlse, losses₄, XY₄, A₄, Z, [1:7 for i=1:10], wlseexp, yidxs=1:7)
    println("Completed test_large()!")
end

test_small()
test_medium()
test_large()