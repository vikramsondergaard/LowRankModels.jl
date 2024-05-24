using LowRankModels, Statistics, Random, CSV, DataFrames, Tables, ArgParse, Dates, Missings
import YAML

export test_pca

function onehot!(df::AbstractDataFrame, col, cate = sort(unique(df[!, col])); outnames = Symbol.(col, cate))
    transform!(df, @. col => ByRow(isequal(cate)) .=> outnames)
    select!(df, Not(col))
    outnames
end

function unonehot!(df::AbstractDataFrame, cols, outname)
    println("cols are $cols")
    transform!(df, AsTable(cols) => ByRow(r -> String(findmax(r)[2])[end] - '0') => outname)
    select!(df, Not(cols))
end

function standardise!(data::DataFrame, rl::Int64)
    for i=1:rl
        data[!, i] = passmissing(convert).(Float64, data[!, i])
        data[!, i] = normalise(data[!, i])
    end
    data
end

function test_pca(test_reg::String)
    args = parse_commandline()

    Random.seed!(1)

    ###### HEY! LOOK AT ME!
    ###############
    # This is a reminder of what you need to do when you get back to your 
    # desk.
    #
    # You're running independence and separation experiments with HSIC on
    # full-sized data. (tabs 3 and 1 respectively, you disorganised fool)
    #
    # Once these are done, `scp` them over to your `adult_full` directory
    # on your laptop. Then run the classifier on them to check their
    # results. Expecting particularly good things from the separation run!

    d = args["data"][1]
    if d == "adult" || d == "adult_low_scale"
        datapath = "data/adult/adult_trimmed.data"
        yamlpath = "data/parameters/adult.yml"
    elseif startswith(d, "ad_observatory")
        cluster = args["cluster"]
        if cluster == 0
            datapath = "data/ad_observatory/WAIST_Data_Only_Interests.csv"
        else
            datapath = "data/ad_observatory/WAIST_Data_Cluster$(cluster).csv"
        end
        yamlpath = "data/parameters/$(d).yml"
    else
        error("Expected one of \"adult\", \"adobservatory\" as a value for `data`, but got $(d)!")
        datapath = ""
    end

    data = CSV.read(datapath, DataFrame, header=1)
    params = YAML.load(open(yamlpath))
    data = dropmissing(data, params["protected_characteristic_idx"])
    rl, bl, cl, ol = parse_losses(params["losses"])
    losses = [rl..., bl..., cl..., ol...]

    cl_start = length(rl) + length(bl) + 1
    cl_end = cl_start + length(cl) - 1
    df_names = names(data)
    cat_names = df_names[cl_start:cl_end]
    for (i, cat_name) in enumerate(cat_names)
        outnames = onehot!(data, cat_name)
        # select!(data, df_names[1:cl_start-2+i], outnames, :)
    end
    df_names = names(data)
    len_ol = length(ol)
    select!(data, df_names[1:cl_start-1], df_names[cl_start+len_ol:end], df_names[cl_start:cl_start+len_ol-1])
    standardise!(data, size(data, 2))
    # display(data)

    s = params["protected_characteristic_idx"]
    k = args["k"]
    y_idx = params["target_feature"]
    
    p = Params(1, max_iter=200, abs_tol=0.0000001, min_stepsize=0.001)

    if d == "adult"
        glrm_pca = pca(convert(Matrix{Float64}, data), args["k"])
    else
        indices = [(i, j) for i=1:size(data, 1), j=1:size(data, 2)]
        obs = filter(x -> !ismissing(data[x[1], x[2]]), indices)
        glrm_pca = pca(passmissing(convert)(Matrix, data), args["k"], obs=obs)
    end
    glrmX, glrmY, ch = fit!(glrm_pca, params=p, verbose=true)
    println("successfully fit vanilla GLRM")

    fairness = args["fairness"][1]

    dir = "data/results/$d/$(args["k"])_components/$test_reg/$(fairness)/benchmarks"
    mkpath(dir)
    fname = "pca_penalty.txt"
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

    # println("Penalty for PCA (without scaling) is $total_orthog")

    data = CSV.read(datapath, DataFrame, header=1)
    standardise!(data, length(rl))
    
    open(fpath, "w") do file
        # penalty = "Fairness penalty (unscaled): $total_orthog"
        penalty = ""
        write(file, "Loss: $(ch.objective[end])\n$penalty")
    end
end