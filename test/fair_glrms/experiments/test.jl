using LowRankModels, Statistics, Random, CSV, DataFrames, Tables, ArgParse, Dates
import YAML

export normalise, parse_commandline, test

normalise(A::AbstractArray) = A .- mean(A)

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "data"
            nargs = 1
            help = "the data set to use (choose from \"adult\", \"adobservatory\")"
            required = true
        "fairness"
            nargs = 1
            help = "which measure of fairness to use"
            required = true
    end
    return parse_args(s)
end

function test(test_reg::String)
    args = parse_commandline()

    Random.seed!(1)

    d = args["data"][1]
    if d == "adult"
        datapath = "/Users/vikramsondergaard/honours/LowRankModels.jl/data/adult/adult_sample.data"
        yamlpath = "/Users/vikramsondergaard/honours/LowRankModels.jl/data/parameters/adult.yml"
    elseif d == "adobservatory"
        datapath = "/path/to/ad_observatory_data"
    else
        error("Expected one of \"adult\", \"adobservatory\" as a value for `data`, but got $(d)!")
        datapath = ""
    end

    data = CSV.read(datapath, DataFrame, header=1)
    params = YAML.load(open(yamlpath))
    rl, bl, cl, ol = parse_losses(params["losses"])

    for i=1:length(rl)
        data[!, i] = convert.(Float64, data[!, i])
        data[!, i] = normalise(data[!, i])
    end

    losses = [rl..., bl..., cl..., ol...]

    s = params["protected_characteristic_idx"]
    k = params["num_reduced_dimensions"]
    y_idx = params["target_feature"]

    m, n = size(data)
    X_init = randn(k, m)
    Y_init = randn(k, embedding_dim(losses))
    groups = partition_groups(data, s, 2)
    weights = [Float64(length(g)) / m for g in groups]
    p = Params(1, max_iter=200, abs_tol=0.0000001, min_stepsize=0.001)

    glrm = GLRM(data, losses, ZeroReg(), ZeroReg(), k)
    glrmX, glrmY, ch = fit!(glrm, params=p, verbose=true)
    println("successfully fit vanilla GLRM")

    fairness = args["fairness"][1]

    println("Starting test for $test_reg using $fairness on the $d dataset at date/time $(now())")

    for scale in params["scales"]
        println("Fitting fair GLRM with scale=$scale")
        if fairness == "hsic"
            regtype = IndependenceReg
        elseif fairness == "orthog"
            regtype = OrthogonalReg
        elseif fairness == "softorthog"
            regtype = SoftOrthogonalReg
        end

        # relative_scale = scale / m
        relative_scale = scale

        if test_reg == "independence"
            regulariser = regtype(relative_scale, normalise(data[:, s]))
        elseif test_reg == "separation"
            regulariser = SeparationReg(relative_scale, data[:, s], data[:, y_idx], regtype)
        elseif test_reg == "sufficiency"
            separator = params["is_target_feature_categorical"] ? encode_to_one_hot(data[:, y_idx]) : data[:, y_idx]
            regulariser = SufficiencyReg(relative_scale, data[:, s], separator, regtype)
        else
            error("Regulariser $test_reg not implemented yet!")
            regulariser = nothing
        end
            
        fglrm = FairGLRM(data, losses, ZeroReg(), ZeroReg(), regulariser, ZeroColReg(), k, s,
            WeightedLogSumExponentialLoss(10^(-6), weights),
            X=copy(X_init), Y=copy(Y_init), Z=groups)
            
        fglrmX, fglrmY, fair_ch = fit!(fglrm, params=p, verbose=true)
        reconstructed = fglrmX' * fglrmY
        fname = "projected_data.csv"
        dir = "data/results/$d/$test_reg/$(fairness)/scale_$scale"
        mkpath(dir)
        fpath = joinpath(dir, fname)
            
        println("successfully fit fair GLRM")
        println("Final loss for this fair GLRM is $(fair_ch.objective[end])")
        CSV.write(fpath, Tables.table(reconstructed))
            
        fair_total_orthog = sum(evaluate(fglrm.rkx[i], fglrmX[i, :]) for i=1:k) / relative_scale
        println("Penalty for fair GLRM (without scaling) is $fair_total_orthog")
        println("Penalty for fair GLRM (with scaling) is $(fair_total_orthog * relative_scale)")
    
        fname = "penalty.txt"
        fpath = joinpath(dir, fname)
    
        total_orthog = sum(evaluate(regulariser, glrmX[i, :]) for i=1:k) / relative_scale
        println("Penalty for vanilla GLRM (without scaling) is $total_orthog")
        println("Penalty for vanilla GLRM (with scaling) is $(total_orthog * relative_scale)")
    
        open(fpath, "w") do file
            write(file, "Loss (Fair GLRM): $(fair_ch.objective[end])\nLoss (Vanilla GLRM): $(ch.objective[end])\nFairness penalty (Fair GLRM): $fair_total_orthog\nFairness penalty (Vanilla GLRM): $total_orthog")
        end
    end

    println("Finished test for $test_reg using $fairness on the $d dataset at date/time $(now())")
    println()

end