using LowRankModels, Statistics, Random, CSV, DataFrames, Tables, ArgParse, Dates
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
        data[!, i] = convert.(Float64, data[!, i])
        data[!, i] = normalise(data[!, i])
    end
    data
end

function test_pca(test_reg::String)
    args = parse_commandline()

    Random.seed!(1)

    d = args["data"][1]
    if d == "adult" || d == "adult_low_scale"
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

    glrm_pca = pca(convert(Matrix{Float64}, data), args["k"])
    glrmX, glrmY, ch = fit!(glrm_pca, params=p, verbose=true)
    println("successfully fit vanilla GLRM")

    fairness = args["fairness"][1]

    dir = "data/results/$d/$(args["k"])_components/$test_reg/$(fairness)/benchmarks"
    mkpath(dir)
    fname = "pca_penalty.txt"
    fpath = joinpath(dir, fname)

    if fairness == "hsic"
        regtype = IndependenceReg
    elseif fairness == "orthog"
        regtype = OrthogonalReg
    elseif fairness == "softorthog"
        regtype = SoftOrthogonalReg
    end

    if test_reg == "independence"
        regulariser = regtype(1.0, normalise(data[:, s]))
    elseif test_reg == "separation"
        regulariser = SeparationReg(1.0, data[:, s], data[:, y_idx], regtype)
    elseif test_reg == "sufficiency"
        separator = params["is_target_feature_categorical"] ? encode_to_one_hot(data[:, y_idx]) : data[:, y_idx]
        regulariser = SufficiencyReg(1.0, data[:, s], separator, regtype)
    else
        error("Regulariser $test_reg not implemented yet!")
        regulariser = nothing
    end

    total_orthog = sum(evaluate(regulariser, glrmX[i, :]) for i=1:k)

    println("Penalty for PCA (without scaling) is $total_orthog")

    data = CSV.read(datapath, DataFrame, header=1)
    standardise!(data, length(rl))

    glrm = GLRM(data, losses, ZeroReg(), ZeroReg(), args["k"])
    loss = objective(glrm, glrmX, glrmY, glrmX' * glrmY)
    
    open(fpath, "w") do file
        write(file, "Loss: $(ch.objective[end])\nFairness penalty (unscaled): $total_orthog")
    end
end