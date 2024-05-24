using LowRankModels, Statistics, Random, CSV, DataFrames, Tables, ArgParse, Dates, CUDA, Missings
import YAML

function test_vanilla_glrm()
    args = parse_commandline()
    println(args)

    Random.seed!(1)

    d = args["data"][1]
    seed = args["seed"][1]
    if d == "adult" || d == "adult_low_scale"
        datapath = "data/adult/splits/$(seed)/original/x_train.csv"
        s_path = "data/adult/splits/$(seed)/original/s_train.csv"
        y_path = "data/adult/splits/$(seed)/original/y_train.csv"
        yamlpath = "data/parameters/$(d).yml"
        savename = "adult/splits/$(seed)"
    elseif d == "adult_test"
        datapath = "data/adult/splits/$(seed)/original/x_test.csv"
        s_path = "data/adult/splits/$(seed)/original/s_test.csv"
        y_path = "data/adult/splits/$(seed)/original/y_test.csv"
        yamlpath = "data/parameters/adult_test.yml"
        savename = "adult/splits/$(seed)"
    elseif d == "celeba"
        datapath = "data/celeba/splits/$(seed)/original/x_train.csv"
        s_path = "data/celeba/splits/$(seed)/original/s_train.csv"
        y_path = "data/celeba/splits/$(seed)/original/y_train.csv"
        savename = "celeba/splits/$(seed)"
        yamlpath = "data/parameters/celeba.yml"
    elseif d == "celeba_test"
        datapath = "data/celeba/splits/$(seed)/original/x_test.csv"
        s_path = "data/celeba/splits/$(seed)/original/s_test.csv"
        y_path = "data/celeba/splits/$(seed)/original/y_test.csv"
        savename = "celeba/splits/$(seed)"
        yamlpath = "data/parameters/celeba.yml"
    elseif d == "toy_data"
        datapath = "data/$(d)/x_train.csv"
        s_path = "data/$(d)/s_train.csv"
        y_path = "data/$(d)/s_train.csv"
        yamlpath = "data/parameters/$(d).yml"
        savename = "toy_data"
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
   #  delete!(data, findall(x -> !x, data["Observation WAIST - Targeted Interests"]))
    rl, bl, cl, ol = parse_losses(params["losses"])

    # standardise!(data, length(rl))

    losses = [rl..., bl..., cl..., ol...]

    k = args["k"]
    
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

    println("The number of columns in the data is $(size(data, 2))")
    println("The number of dimensions in the losses is $(embedding_dim(losses))")
    if startswith(d, "adult") || startswith(d, "celeba")
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

    fname = d == "adult_test" ? "x_test.csv" : "x_train.csv"
    dir = "data/$(savename)/results/$(k)_components"
    mkpath(dir)
    xname = d == "adult_test" ? "vanilla_glrmX_test.csv" : "vanilla_glrmX.csv"
    yname = d == "adult_test" ? "vanilla_glrmY_test.csv" : "vanilla_glrmY.csv"
    xpath = joinpath(dir, xname)
    ypath = joinpath(dir, yname)
    CSV.write(xpath, Tables.table(glrmX))
    CSV.write(ypath, Tables.table(glrmY))

    println("Resulting shape of glrmY is $(size(glrmY))")

    return glrmX, glrmY
end