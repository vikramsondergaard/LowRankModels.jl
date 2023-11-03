using LowRankModels, Statistics, Random, CSV, DataFrames, Tables

Random.seed!(1)

scales = [10^(-6), 10^(-5), 3 * 10^(-3), 10^(-2), 10^(-1), 5 * 10^(-1), 1, 5,
          10, 20, 60, 10^2, 10^3, 10^4, 10^5]

A₂_bool = [true, false, true]
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

function test(A::AbstractArray, losses::Array{Loss, 1}, s::Int64, k::Int64,
        r::ColumnRegularizer)
    m, n = size(A)
    X_init = randn(k, m)
    Y_init = randn(k, embedding_dim(losses))
    groups = partition_groups(A, s, 2)
    weights = [Float64(length(g)) / m for g in groups]
    p = Params(1, max_iter=200, abs_tol=0.0000001, min_stepsize=0.001)
        
    for scale in scales
        println("Fitting independent fair GLRM with scale=$scale")
        
        fglrm = FairGLRM(A, losses, IndependenceReg(scale, A[:, s]), ZeroReg(), k, s,
            WeightedLogSumExponentialLoss(10^(-6), weights),
            X=X_init, Y=Y_init, Z=groups)
        
        fglrmX, fglrmY, ch = fit!(fglrm, params=p, verbose=true)
        reconstructed = fglrmX' * fglrmY
        fname = "test_adult_independence_hsic_scale_$(scale).csv"
        dir = "data/results/reconstructed"
        fpath = joinpath(dir, fname)
        
        println("successfully fit fair GLRM")
        println("Final loss for this fair GLRM is $(ch.objective[end])")
        CSV.write(fpath, Tables.table(reconstructed), ',')
        
        total_orthog = sum(evaluate(fglrm.rx[1], fglrmX[k, :]) for k=1:fglrm.k) / scale
        println("Independence penalty (without scaling) is $total_orthog")

        fname = "penalty_adult_independence_hsic_scale_$(scale).txt"
        dir = "data/results/penalties"
        fpath = joinpath(dir, fname)
        open(fpath, "w") do file
            write(file, string(total_orthog))
        end
    end
end

function test_small()
    # TODO implement this test
    A₂ = Any[A₂_bool A₂_real]
    losses₂ = [HingeLoss(), QuadLoss()]

    k = 1
    s = 1

    test(A₂, losses₂, s, k)

    p = Params(1, max_iter=200, abs_tol=0.0000001, min_stepsize=0.001)
    glrm = GLRM(A₂, losses₂, ZeroReg(), ZeroReg(), k)
    glrmX, glrmY, ch = fit!(glrm, params=p, verbose=false)
    println("successfully fit vanilla GLRM")
    total_orthog = sum(evaluate(IndependenceReg(1.0, A₂[:, s]), glrmX[i, :]) for i=1:k)
    println("Independence penalty (without scaling) is $total_orthog")

    println("Passed test_small()!")
end

function test_medium()
    # TODO implement this test
    A₃ = Any[A₃_cat A₃_bool A₃_real A₃_ord]
    losses₃ = [OvALoss(3, bin_loss=HingeLoss()), HingeLoss(),
        QuadLoss(), OrdinalHingeLoss(4)]
    
    k = 2
    s = 2
    
    test(A₃, losses₃, s, k)

    p = Params(1, max_iter=200, abs_tol=0.0000001, min_stepsize=0.001)
    glrm = GLRM(A₃, losses₃, ZeroReg(), ZeroReg(), k)
    glrmX, glrmY, ch = fit!(glrm, params=p, verbose=false)
    println("successfully fit vanilla GLRM")
    total_orthog = sum(evaluate(IndependenceReg(1.0, A₃[:, s]), glrmX[i, :]) for i=1:k)
    println("Independence penalty (without scaling) is $total_orthog")

    println("Passed test_medium()!")
end

function test_large()
    # TODO implement this test
    A₄ = Any[A₄_cat A₄_bool A₄_real A₄_ord]
    losses₄ = [OvALoss(4, bin_loss=HingeLoss()),
               OvALoss(6, bin_loss=HingeLoss()), HingeLoss(), QuadLoss(),
               QuadLoss(), OrdinalHingeLoss(3), OrdinalHingeLoss(5)]

    k = 3
    s = 3

    test(A₄, losses₄, s, k)

    p = Params(1, max_iter=200, abs_tol=0.0000001, min_stepsize=0.001)
    glrm = GLRM(A₄, losses₄, ZeroReg(), ZeroReg(), k)
    glrmX, glrmY, ch = fit!(glrm, params=p, verbose=false)
    println("successfully fit vanilla GLRM")
    total_orthog = sum(evaluate(IndependenceReg(1.0, A₄[:, s]), glrmX[i, :]) for i=1:k)
    println("Independence penalty (without scaling) is $total_orthog")

    println("Passed test_large()!")
end

function test_adult()
    # Path to the Adult dataset
    # datapath = "/Users/vikramsondergaard/honours/data/adult/adult_trimmed.data"
    datapath = "/Users/vikramsondergaard/honours/LowRankModels.jl/data/adult/adult_sample.data"
    # Read the CSV file and convert it to a matrix
    adult_csv = CSV.read(datapath, DataFrame, header=1)
    A = convert(Matrix, adult_csv)
    # Gender is the 10th column in the Adult data set
    s = 3
    k = 2

    real_losses = [QuadLoss(), QuadLoss()]

    bool_losses = [HingeLoss(), HingeLoss()]

    n_workclasses = 9
    n_relationships = 6
    n_races = 5 # (from the data - there are not five races)
    n_educations = 16
    cat_losses = [OvALoss(n_workclasses, bin_loss=HingeLoss()),
                OvALoss(n_relationships, bin_loss=HingeLoss()),
                OvALoss(n_races, bin_loss=HingeLoss())]

    ord_losses = [OrdinalHingeLoss(n_educations)]

    losses = [real_losses..., bool_losses..., cat_losses..., ord_losses...]

    m, n = size(A)
    X_init = randn(k, m)
    Y_init = randn(k, embedding_dim(losses))
    groups = partition_groups(A, s, 2)
    weights = [Float64(length(g)) / m for g in groups]
    p = Params(1, max_iter=200, abs_tol=0.0000001, min_stepsize=0.001)

    glrm = GLRM(A, losses, ZeroReg(), ZeroReg(), k)
    glrmX, glrmY, ch = fit!(glrm, params=p, verbose=false)
    println("successfully fit vanilla GLRM")
        
    for scale in scales
        println("Fitting independent fair GLRM with scale=$scale")
        
        fglrm = FairGLRM(A, losses, ZeroReg(), ZeroReg(), IndependenceReg(scale, A[:, s]), ZeroColReg(), k, s,
            WeightedLogSumExponentialLoss(10^(-6), weights),
            X=X_init, Y=Y_init, Z=groups)
        
        fglrmX, fglrmY, ch = fit!(fglrm, params=p, verbose=true)
        reconstructed = fglrmX' * fglrmY
        fname = "test_adult_independence_hsic_scale_$(scale).csv"
        dir = "data/results/reconstructed"
        fpath = joinpath(dir, fname)
        
        println("successfully fit fair GLRM")
        println("Final loss for this fair GLRM is $(ch.objective[end])")
        CSV.write(fpath, Tables.table(reconstructed))
        
        fair_total_orthog = sum(evaluate(fglrm.rkx[i], fglrmX[i, :]) for i=1:k) / scale
        println("Independence penalty (without scaling) is $fair_total_orthog")

        fname = "penalty_adult_independence_hsic_scale_$(scale).txt"
        dir = "data/results/penalties"
        fpath = joinpath(dir, fname)

        total_orthog = sum(evaluate(IndependenceReg(scale, A[:, s]), glrmX[i, :]) for i=1:k) / scale
        println("Independence penalty (without scaling) is $total_orthog")

        open(fpath, "w") do file
            write(file, "Fair GLRM: $fair_total_orthog\nVanilla GLRM: $total_orthog")
        end
    end

    # p = Params(1, max_iter=200, abs_tol=0.0000001, min_stepsize=0.001)
    # glrm = GLRM(A, losses, ZeroReg(), ZeroReg(), k)
    # glrmX, glrmY, ch = fit!(glrm, params=p, verbose=false)
    # println("successfully fit vanilla GLRM")
    # total_orthog = sum(evaluate(SoftOrthogonalReg(scale, A[:, s]), glrmX[i, :]) for i=1:k)
    # println("Independence penalty (without scaling) is $total_orthog")

    println("Passed test_large()!")
end

function test_adult_orthog()
    # Path to the Adult dataset
    # datapath = "/Users/vikramsondergaard/honours/data/adult/adult_trimmed.data"
    datapath = "/Users/vikramsondergaard/honours/LowRankModels.jl/data/adult/adult_sample.data"
    # Read the CSV file and convert it to a matrix
    adult_csv = CSV.read(datapath, DataFrame, header=1)
    A = convert(Matrix, adult_csv)
    # Gender is the 10th column in the Adult data set
    s = 3
    k = 2

    real_losses = [QuadLoss(), QuadLoss()]

    bool_losses = [HingeLoss(), HingeLoss()]

    n_workclasses = 9
    n_relationships = 6
    n_races = 5 # (from the data - there are not five races)
    n_educations = 16
    cat_losses = [OvALoss(n_workclasses, bin_loss=HingeLoss()),
                OvALoss(n_relationships, bin_loss=HingeLoss()),
                OvALoss(n_races, bin_loss=HingeLoss())]

    ord_losses = [OrdinalHingeLoss(n_educations)]

    losses = [real_losses..., bool_losses..., cat_losses..., ord_losses...]

    m, n = size(A)
    X_init = randn(k, m)
    Y_init = randn(k, embedding_dim(losses))
    groups = partition_groups(A, s, 2)
    weights = [Float64(length(g)) / m for g in groups]
    p = Params(1, max_iter=200, abs_tol=0.0000001, min_stepsize=0.001)
        
        
    fglrm = FairGLRM(A, losses, ZeroReg(), ZeroReg(), OrthogonalReg(A[:, s]), ZeroColReg(), k, s,
        WeightedLogSumExponentialLoss(10^(-6), weights),
        X=X_init, Y=Y_init, Z=groups)
        
    fglrmX, fglrmY, ch = fit!(fglrm, params=p, verbose=true)
    reconstructed = fglrmX' * fglrmY
    fname = "test_adult_independence_orthog.csv"
    dir = "data/results/reconstructed"
    fpath = joinpath(dir, fname)
        
    println("successfully fit fair GLRM")
    println("Final loss for this fair GLRM is $(ch.objective[end])")
    CSV.write(fpath, Tables.table(reconstructed))
        
    fair_total_orthog = sum(evaluate(fglrm.rkx[i], fglrmX[i, :]) for i=1:fglrm.k)
    println("Independence penalty (without scaling) is $fair_total_orthog")


    p = Params(1, max_iter=200, abs_tol=0.0000001, min_stepsize=0.001)
    glrm = GLRM(A, losses, ZeroReg(), ZeroReg(), k)
    glrmX, glrmY, ch = fit!(glrm, params=p, verbose=false)
    println("successfully fit vanilla GLRM")
    total_orthog = sum(evaluate(OrthogonalReg(1.0, A[:, s]), glrmX[i, :]) for i=1:k)
    println("Independence penalty (without scaling) is $total_orthog")

    fname = "penalty_adult_independence_orthog.txt"
    dir = "data/results/penalties"
    fpath = joinpath(dir, fname)
    open(fpath, "w") do file
        write(file, "Fair GLRM: $fair_total_orthog\nVanilla GLRM: $total_orthog")
    end

    println("Passed test_large()!")
end

function test_adult_softorthog()
    # Path to the Adult dataset
    # datapath = "/Users/vikramsondergaard/honours/data/adult/adult_trimmed.data"
    datapath = "/Users/vikramsondergaard/honours/LowRankModels.jl/data/adult/adult_sample.data"
    # Read the CSV file and convert it to a matrix
    adult_csv = CSV.read(datapath, DataFrame, header=1)
    A = convert(Matrix, adult_csv)
    # Gender is the 10th column in the Adult data set
    s = 3
    k = 2

    real_losses = [QuadLoss(), QuadLoss()]

    bool_losses = [HingeLoss(), HingeLoss()]

    n_workclasses = 9
    n_relationships = 6
    n_races = 5 # (from the data - there are not five races)
    n_educations = 16
    cat_losses = [OvALoss(n_workclasses, bin_loss=HingeLoss()),
                OvALoss(n_relationships, bin_loss=HingeLoss()),
                OvALoss(n_races, bin_loss=HingeLoss())]

    ord_losses = [OrdinalHingeLoss(n_educations)]

    losses = [real_losses..., bool_losses..., cat_losses..., ord_losses...]

    m, n = size(A)
    X_init = randn(k, m)
    Y_init = randn(k, embedding_dim(losses))
    groups = partition_groups(A, s, 2)
    weights = [Float64(length(g)) / m for g in groups]
    p = Params(1, max_iter=200, abs_tol=0.0000001, min_stepsize=0.001)

    glrm = GLRM(A, losses, ZeroReg(), ZeroReg(), k)
    glrmX, glrmY, ch = fit!(glrm, params=p, verbose=false)
    println("successfully fit vanilla GLRM")
        
    for scale in scales
        println("Fitting independent fair GLRM with scale=$scale")
        
        fglrm = FairGLRM(A, losses, ZeroReg(), ZeroReg(), SoftOrthogonalReg(scale, A[:, s]), ZeroColReg(), k, s,
            WeightedLogSumExponentialLoss(10^(-6), weights),
            X=X_init, Y=Y_init, Z=groups)
        
        fglrmX, fglrmY, ch = fit!(fglrm, params=p, verbose=true)
        reconstructed = fglrmX' * fglrmY
        fname = "test_adult_independence_softorthog_scale_$(scale).csv"
        dir = "data/results/reconstructed"
        fpath = joinpath(dir, fname)
        
        println("successfully fit fair GLRM")
        println("Final loss for this fair GLRM is $(ch.objective[end])")
        CSV.write(fpath, Tables.table(reconstructed))
        
        fair_total_orthog = sum(evaluate(fglrm.rkx[i], fglrmX[i, :]) for i=1:k) / scale
        println("Independence penalty (without scaling) is $fair_total_orthog")

        fname = "penalty_adult_independence_softorthog_scale_$(scale).txt"
        dir = "data/results/penalties"
        fpath = joinpath(dir, fname)

        total_orthog = sum(evaluate(SoftOrthogonalReg(scale, A[:, s]), glrmX[i, :]) for i=1:k) / scale
        println("Independence penalty (without scaling) is $total_orthog")

        open(fpath, "w") do file
            write(file, "Fair GLRM: $fair_total_orthog\nVanilla GLRM: $total_orthog")
        end
    end

    # p = Params(1, max_iter=200, abs_tol=0.0000001, min_stepsize=0.001)
    # glrm = GLRM(A, losses, ZeroReg(), ZeroReg(), k)
    # glrmX, glrmY, ch = fit!(glrm, params=p, verbose=false)
    # println("successfully fit vanilla GLRM")
    # total_orthog = sum(evaluate(SoftOrthogonalReg(scale, A[:, s]), glrmX[i, :]) for i=1:k)
    # println("Independence penalty (without scaling) is $total_orthog")

    println("Passed test_large()!")
end

# test_small()
# println()
# test_medium()
# println()
# test_large()

test_adult()
# test_adult_orthog()
# test_adult_softorthog()