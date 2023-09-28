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

function test(l::WeightedLogSumExponentialLoss, i, j, losses::Array{Loss, 1},
        XY, A, Z, observed_features, gradexp::Float64; 
        yidxs = get_yidxs(losses), refresh = (i * j == 1))
    gradout = grad(l, i, j, losses, XY, A, Z, observed_features, yidxs=yidxs, refresh=refresh)
    println("Expected: $gradexp")
    println("Got: $gradout")
    @assert isapprox(gradexp, gradout, atol=0.0001)
end

function test_small()
    wlse = WeightedLogSumExponentialLoss(10^(-6), [0.5, 0.5])
    Z = [[1, 3], [2]]
    
end