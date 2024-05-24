using LowRankModels, Statistics, Random, CSV, DataFrames, Tables, ArgParse, Dates, CUDA, Profile, Missings
import YAML

export normalise, parse_commandline, test, standardise!

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
        "seed"
            nargs = 1
            help = "the random seed with which to split the data"
            required = true
            arg_type = Int
            range_tester = x -> (x >= 0 && x < 10)
        "-k"
            arg_type = Int
            help = "the number of components to which to reduce the data"
            default = 2
        "-s", "--scales"
            nargs = '+'
            arg_type = Float64
            help = "the scales to use for experiments (if this isn't provided, uses a corresponding YAML file instead)"
            required = false
        "-x", "--startX"
            nargs = 1
            arg_type = String
            required = false
        "-y", "--startY"
            nargs = 1
            arg_type = String
            required = false
        "-g", "--gpu"
            nargs = 0
            help = "whether to use the GPU or not (will cause errors if you don't have a CUDA driver)"
            action = :store_true
        "-c", "--cluster"
            arg_type = Int
            help = "(for ad observatory data) whether to use a certain WAIST interest data set or not"
            default = 0
        "-w", "--weights"
            nargs = '+'
            arg_type = Float64
            help = "the weights used for Buet-Golfouse and Utyagulov's fair GLRM (2022) (if this isn't provided, just splits weights evenly, recreating vanilla GLRM)"
            required = false
        "-a", "--alpha"
            arg_type = Float64
            help = "the degree of group-based fairness to include (from Buet-Golfouse and Utyagulov (2022))"
            default = 1.0e-6
            required = false
    end
    return parse_args(s)
end

function standardise!(data::DataFrame, rl::Int64)
    for i=1:rl
        data[!, i] = convert.(Float64, data[!, i])
        data[!, i] = normalise(data[!, i])
    end
    data
end

function test(test_reg::String)
    args = parse_commandline()

    Random.seed!(1)

    d = args["data"][1]
    cluster = args["cluster"]
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
    elseif startswith(d, "ad_observatory")
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

    test(test_reg, randn(args["k"], size(data, 1)), randn(args["k"], embedding_dim(losses)))
end

function group(data, s)
    data_copy = hcat(data, s)
    data_copy[!, :row_idx] = 1:size(data_copy, 1)
    gdf = groupby(data_copy, names(s), sort=true)
    groups = [g[:row_idx] for g in gdf]
    groups
end

function test(test_reg::String, glrmX::AbstractArray, glrmY::AbstractArray)
    args = parse_commandline()
    println(args)
    println("Number of threads is $(Threads.nthreads())")

    Random.seed!(1)

    d = args["data"][1]
    cluster = args["cluster"]
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
    elseif startswith(d, "ad_observatory")
        if cluster == 0
            datapath = "data/ad_observatory/WAIST_Data_Only_Interests.csv"
        else
            datapath = "data/ad_observatory/WAIST_Data_Cluster$(cluster).csv"
        end
        yamlpath = "data/parameters/$(d).yml"
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

    k = args["k"]

    m, n = size(data)
    groups = group(data, s)
    p = Params(1, max_iter=200, abs_tol=0.0000001, min_stepsize=0.001)

    fairness = args["fairness"][1]

    println("Starting test for $test_reg using $fairness on the $d dataset at date/time $(now())")

    scales = isempty(args["scales"]) ? params["scales"] : args["scales"]
    weights = isempty(args["weights"]) ? [Float64(length(g)) / m for g in groups] : args["weights"]

    for scale in scales
        println("Fitting fair GLRM with scale=$scale")
        glrm_scale = 1.0 - scale
        for l in losses
            mul!(l, glrm_scale)
        end
        if fairness == "hsic" || fairness == "equality_of_opportunity"
            regtype = HSICReg
            relative_scale = scale
        elseif fairness == "orthog"
            regtype = OrthogonalReg
            relative_scale = scale
        elseif fairness == "softorthog"
            regtype = SoftOrthogonalReg
            relative_scale = scale / m
        end
        relative_scale = scale

        if test_reg == "nothing"
            regulariser = ZeroColReg()
        elseif fairness == "equality_of_opportunity"
            target_groups = group(data, y)[2]
            regulariser = [SeparationReg(relative_scale, convert(Array, s), target_groups, regtype, get_nfsic(CuArray(convert(Array, s)), CuArray(glrmX[i, :]))) for i=1:k]
        elseif test_reg == "independence"
            if fairness == "hsic"
                regulariser = [regtype(relative_scale, convert(Array, s), glrmX[i, :], NFSIC) for i=1:k]
            else
                regulariser = regtype(relative_scale, convert(Array, s))
            end
        elseif test_reg == "separation"
            if fairness == "hsic"
                regulariser = [SeparationReg(relative_scale, convert(Array, s), convert(Array, y), regtype, get_nfsic(CuArray(convert(Array, s)), (CuArray(glrmX[i, :])))) for i=1:k]
            else
                regulariser = SeparationReg(relative_scale, convert(Array, s), convert(Array, y), regtype)
            end
        elseif test_reg == "sufficiency"
            separator = params["is_target_feature_categorical"] ? encode_to_one_hot(convert(Array, y)) : convert(Array, y)
            if fairness == "hsic"
                regulariser = [SufficiencyReg(relative_scale, convert(Array, s), separator, regtype, get_nfsic(CuArray(convert(Array, s)), CuArray(glrmX[i, :]))) for i=1:k]
            else
                regulariser = SufficiencyReg(relative_scale, convert(Array, s), separator, regtype)
            end
        else
            error("Regulariser $test_reg not implemented yet!")
            regulariser = nothing
        end
            
        alpha = args["alpha"]
        if startswith(d, "adult") || d == "toy_data" || startswith(d, "celeba")
            fglrm = FairGLRM(data, losses, ZeroReg(), ZeroReg(), regulariser, ZeroColReg(), k, s,
                WeightedLogSumExponentialLoss(alpha, weights),
                X=glrmX, Y=glrmY, Z=groups)
        elseif startswith(d, "ad_observatory")
            indices = [(i, j) for i=1:size(data, 1), j=1:size(data, 2)]
            obs = filter(x -> !ismissing(data[x[1], x[2]]), indices)
            fglrm = FairGLRM(data, losses, ZeroReg(), ZeroReg(), regulariser, ZeroColReg(), k, s,
                WeightedLogSumExponentialLoss(alpha, weights),
                X=glrmX, Y=glrnY, Z=groups, obs=obs)
        end

        dir = "data/$(savename)/results/$(test_reg)/$(k)_components/scale_$(relative_scale)/$(fairness)"
        ch = ConvergenceHistory(dir)    
        fglrmX, fglrmY, fair_ch = fit!(fglrm, params=p, ch=ch, verbose=true, checkpoint=true)
        reconstructed = fglrmX' * fglrmY
        fname = d == "adult_test" ? "x_test.csv" : "x_train.csv"
        clustername = cluster > 0 ? "cluster_$(cluster)" : "no_interests"
        mkpath(dir)
        fpath = joinpath(dir, fname)
            
        println("successfully fit fair GLRM")
        println("Final loss for this fair GLRM is $(fair_ch.objective[end])")
        CSV.write(fpath, Tables.table(reconstructed))
    
        fname = "penalty.txt"
        fpath = joinpath(dir, fname)

        if fairness == "orthog" break end
        for l in losses
            mul!(l, 1.0 / glrm_scale)
        end
    end

    println("Finished test for $test_reg using $fairness on the $d dataset at date/time $(now())")
    println()

end