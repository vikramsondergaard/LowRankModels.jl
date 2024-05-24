using LowRankModels, Statistics, Random, CSV, DataFrames, Tables, ArgParse, Dates, Missings, JSON
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

    d = args["data"][1]
    seed = args["seed"][1]
    if d == "adult" || d == "adult_low_scale"
        datapath = "data/adult/splits/$(seed)/original/x_train.csv"
        s_path = "data/adult/splits/$(seed)/original/s_train.csv"
        y_path = "data/adult/splits/$(seed)/original/y_train.csv"
        yamlpath = "data/parameters/$(d).yml"
    elseif d == "adult_test"
        datapath = "data/adult/splits/$(seed)/original/x_test.csv"
        s_path = "data/adult/splits/$(seed)/original/s_test.csv"
        y_path = "data/adult/splits/$(seed)/original/y_test.csv"
        yamlpath = "data/parameters/adult_test.yml"
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
    s = CSV.read(s_path, DataFrame, header=1)
    y = CSV.read(y_path, DataFrame, header=1)
    # data = dropmissing(data, params["protected_characteristic_idx"])
    missing_indices = Set{Int}()
    for name in names(s)
        inds = findall(ismissing, s[:, name])
        for ind in inds push!(missing_indices, ind) end
    end
    # Need to drop indices of the data that are missing in the protected characteristic
    delete!(data, sort(collect(missing_indices)))
    dropmissing!(s)
    rl, bl, cl, ol = parse_losses(params["losses"])
    losses = [rl..., bl..., cl..., ol...]

    cl_start = length(rl) + length(bl) + 1
    cl_end = cl_start + length(cl) - 1
    df_names = names(data)
    cat_names = df_names[cl_start:cl_end]

    df_names = names(data)
    len_ol = length(ol)

    k = args["k"]
    
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
    
    open(fpath, "w") do file
        # penalty = "Fairness penalty (unscaled): $total_orthog"
        penalty = ""
        write(file, "Loss: $(ch.objective[end])\n$penalty")
    end
end