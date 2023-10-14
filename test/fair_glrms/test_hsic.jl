using LowRankModels, Statistics

scales = [10^(-6), 10^(-5), 3 * 10^(-3), 10^(-2), 10^(-1), 5 * 10^(-1), 1, 5,
          10, 20, 60, 10^2, 10^3, 10^4, 10^5]

A₂_bool = [1, -1, 1]
A₂_real = [-0.82745, 1.26234, -0.29391]

A₃_cat = [1, 1, 1, 2, 2, 3]
A₃_bool = [false, false, true, true, false, false]
A₃_real = [0.49515, -0.15264, 0.77662, -1.65449, -0.17560, 0.90025]
A₃_ord = [1, 2, 1, 4, 4, 2]

A₄_cat = [1 6
          1 4
          1 5
          1 2
          2 3
          2 2
          2 1
          3 5
          3 1
          4 3]
A₄_bool = [false, false, true, true, false, false, true, false, false, false]
A₄_real = [-0.83625 -0.62522
            1.62911  0.62544
            1.19634 -0.67659
           -0.19417  0.05815
           -0.54548 -0.25093
            0.20709 -0.73999
            1.05964 -1.04975
           -0.41204  0.05053
            0.80596  0.56225
            0.28682 -0.57457]
A₄_ord = [2 3
          3 3
          2 4
          2 3
          2 3
          1 1
          1 2
          2 4
          2 5
          1 1]

# TODO implement unit tests below
# General idea is:
# - For every scale in scales, define HSIC regualisers over the columns of X
#   with that scale
# - Fit a fair GLRM using just these regualisers (zero regularisers everywhere)
# - Measure HSIC between columns of X and protected characteristic to see if it
#   significantly (could even use the threshold here) enforces independence
# - Find the scale(s) that best enforce independence for the data

function test_small()
    # TODO implement this test
    println("NOT YET IMPLEMENTED")
    A₂ = Any[A₂_bool A₂_real]
    losses₂ = [HingeLoss(), QuadLoss()]

    println("Passed test_small()!")
end

function test_medium()
    # TODO implement this test
    println("NOT YET IMPLEMENTED")
    A₃ = Any[A₃_cat A₃_bool A₃_real A₃_ord]
    losses₃ = [OvALoss(3, bin_loss=HingeLoss()), HingeLoss(),
        QuadLoss(), OrdinalHingeLoss(4)]

    println("Passed test_medium()!")
end

function test_large()
    # TODO implement this test
    println("NOT YET IMPLEMENTED")
    A₄ = Any[A₄_cat A₄_bool A₄_real A₄_ord]
    losses₄ = [OvALoss(4, bin_loss=HingeLoss()),
               OvALoss(6, bin_loss=HingeLoss()), HingeLoss(), QuadLoss(),
               QuadLoss(), OrdinalHingeLoss(3), OrdinalHingeLoss(5)]

    println("Passed test_large()!")
end