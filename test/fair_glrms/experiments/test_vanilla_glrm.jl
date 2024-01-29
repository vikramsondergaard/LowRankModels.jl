using LowRankModels, Statistics, Random, CSV, DataFrames, Tables, ArgParse, Dates, CUDA, Missings
import YAML

function test_vanilla_glrm(test_reg::String)
    args = parse_commandline()
    println(args)

    Random.seed!(1)

    d = args["data"][1]
    if d == "adult" || d == "adult_low_scale"
        datapath = "data/adult/adult_trimmed.data"
        yamlpath = "data/parameters/adult.yml"
    elseif d == "adobservatory"
        datapath = "data/ad_observatory/WAIST_Data_No_Interests.csv"
        yamlpath = "data/parameters/ad_observatory_no_interests.yml"
    else
        error("Expected one of \"adult\", \"adobservatory\" as a value for `data`, but got $(d)!")
        datapath = ""
    end

    data = CSV.read(datapath, DataFrame, header=1)
    params = YAML.load(open(yamlpath))
    if typeof(params["protected_characteristic_idx"]) <: Array
        for col in params["protected_characteristic_idx"]
            deleterows!(data, findall(ismissing, data[:, col]))
        end
    else
        deleterows!(data, findall(ismissing, data[:, params["protected_characteristic_idx"]]))
    end
    rl, bl, cl, ol = parse_losses(params["losses"])

    # standardise!(data, length(rl))

    losses = [rl..., bl..., cl..., ol...]

    s = params["protected_characteristic_idx"]
    k = args["k"]
    y_idx = params["target_feature"]
    
    p = Params(1, max_iter=200, abs_tol=0.0000001, min_stepsize=0.001)
    # X_init = randn(Float64, k, size(data, 1))
    X_init = randn(Float64, k, size(data, 1))
    Y_init = randn(Float64, k, embedding_dim(losses))

    # if args["gpu"]
    #     X_init = CUDA.randn(Float64, k, size(data, 1))
    #     Y_init = CUDA.randn(Float64, k, embedding_dim(losses))
    # else
    #     X_init = randn(Float64, k, size(data, 1))
    #     Y_init = randn(Float64, k, size(data, 2))
    # end

    if d == "adult"
        glrm = GLRM(data, losses, ZeroReg(), ZeroReg(), k; X=X_init, Y=Y_init)
    else
        indices = [(i, j) for i=1:size(data, 1), j=1:size(data, 2)]
        obs = filter(x -> !ismissing(data[x[1], x[2]]), indices)
        glrm = GLRM(data, losses, ZeroReg(), ZeroReg(), k; X=X_init, Y=Y_init,
            obs=obs)
    end
    glrmX, glrmY, ch = fit!(glrm, params=p, verbose=true)
    println("successfully fit vanilla GLRM")

    fairness = args["fairness"][1]

    dir = "data/results/$d/$(args["k"])_components/$test_reg/$(fairness)/benchmarks"
    mkpath(dir)
    fname = "vanilla_glrm_penalty.txt"
    fpath = joinpath(dir, fname)

    # if fairness == "hsic"
    #     regtype = HSICReg
    # elseif fairness == "orthog"
    #     regtype = OrthogonalReg
    # elseif fairness == "softorthog"
    #     regtype = SoftOrthogonalReg
    # end

    # if test_reg == "independence"
    #     if typeof(params["protected_characteristic_idx"]) <: Array
    #         regulariser = regtype(1.0, normalise(convert(Matrix, data[:, s])))
    #     else
    #         regulariser = regtype(1.0, normalise(data[:, s]))
    #     end
    # elseif test_reg == "separation"
    #     regulariser = SeparationReg(1.0, data[:, s], data[:, y_idx], regtype)
    # elseif test_reg == "sufficiency"
    #     separator = params["is_target_feature_categorical"] ? encode_to_one_hot(data[:, y_idx]) : data[:, y_idx]
    #     regulariser = SufficiencyReg(1.0, data[:, s], separator, regtype)
    # else
    #     error("Regulariser $test_reg not implemented yet!")
    #     regulariser = nothing
    # end

    # total_orthog = sum(evaluate(regulariser, glrmX[i, :]) for i=1:k)

    # println("Penalty for vanilla GLRM (without scaling) is $total_orthog")
    
    open(fpath, "w") do file
        # penalty = "Fairness penalty (unscaled): $total_orthog"
        penalty = ""
        write(file, "Loss: $(ch.objective[end])\n$penalty")
    end

    println("Resulting shape of glrmY is $(size(glrmY))")

    return glrmX, glrmY
end