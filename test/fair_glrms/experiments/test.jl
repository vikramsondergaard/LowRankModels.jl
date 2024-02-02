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
        "-k"
            arg_type = Int
            help = "the number of components to which to reduce the data"
            default = 2
        "-s", "--scales"
            nargs = '+'
            arg_type = Float64
            help = "the scales to use for experiments (if this isn't provided, uses a corresponding YAML file instead)"
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
    if d == "adult" || d == "adult_low_scale"
        datapath = "data/adult/adult_trimmed.data"
        yamlpath = "data/parameters/$(d).yml"
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
    data = dropmissing(data, params["protected_characteristic_idx"])

    rl, bl, cl, ol = parse_losses(params["losses"])
    losses = [rl..., bl..., cl..., ol...]

    test(test_reg, randn(args["k"], size(data, 1)), randn(args["k"], embedding_dim(losses)))
end

function test(test_reg::String, glrmX::AbstractArray, glrmY::AbstractArray)
    args = parse_commandline()
    println(args)
    println("Number of threads is $(Threads.nthreads())")

    Random.seed!(1)

    d = args["data"][1]
    cluster = args["cluster"]
    if d == "adult" || d == "adult_low_scale"
        datapath = "data/adult/adult_trimmed.data"
        yamlpath = "data/parameters/$(d).yml"
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
    data = dropmissing(data, params["protected_characteristic_idx"])
    # delete!(data, findall(x -> !x, data["Observation WAIST - Targeted Interests"]))
    # print("Does the data now contain missing values?")
    # for col in params["protected_characteristic_idx"]
    #     print("The missing indices in $col are $(findall(ismissing, data[:, col]))")
    # end
    rl, bl, cl, ol = parse_losses(params["losses"])

    standardise!(data, length(rl))

    losses = [rl..., bl..., cl..., ol...]

    s = params["protected_characteristic_idx"]
    k = args["k"]
    y_idx = params["target_feature"]

    m, n = size(data)
    if typeof(s) <: Array
        data_copy = copy(data)
        data_copy[!, :row_idx] = 1:size(data_copy, 1)
        gdf = groupby(data_copy, [names(data_copy)[i] for i in s])
        groups = [g[:row_idx] for g in gdf]
        data_copy = nothing # clean it up afterwards so it doesn't take up space
    else
        data_copy = copy(data)
        data_copy[!, :row_idx] = 1:size(data_copy, 1)
        gdf = groupby(data_copy, names(data_copy)[s])
        groups = [g[:row_idx] for g in gdf]
        data_copy = nothing
    end
    # weights = [Float64(length(g)) / m for g in groups]
    # println("weights are: $weights")
    p = Params(1, max_iter=200, abs_tol=0.0000001, min_stepsize=0.001)

    fairness = args["fairness"][1]

    println("Starting test for $test_reg using $fairness on the $d dataset at date/time $(now())")

    scales = isempty(args["scales"]) ? params["scales"] : args["scales"]
    weights = isempty(args["weights"]) ? [Float64(length(g)) / m for g in groups] : args["weights"]

    for scale in scales
        println("Fitting fair GLRM with scale=$scale")
        if fairness == "hsic"
            regtype = HSICReg
            relative_scale = scale
        elseif fairness == "orthog"
            regtype = OrthogonalReg
            relative_scale = scale
        elseif fairness == "softorthog"
            regtype = SoftOrthogonalReg
            relative_scale = scale / m
        end

        # relative_scale = scale / m

        if test_reg == "nothing"
            regulariser = ZeroColReg()
        elseif test_reg == "independence"
            if fairness == "hsic"
                if typeof(params["protected_characteristic_idx"]) <: Array
                    regulariser = [regtype(relative_scale, Float64.(convert(Matrix, data[:, s])), glrmX[i, :], NFSIC) for i=1:k]
                else
                    regulariser = [regtype(relative_scale, Float64.(data[:, s]), glrmX[i, :], NFSIC) for i=1:k]
                end
            else
                regulariser = regtype(relative_scale, normalise(data[:, s]))
            end
        elseif test_reg == "separation"
            if fairness == "hsic"
                if typeof(params["protected_characteristic_idx"]) <: Array
                    regulariser = [SeparationReg(relative_scale, convert(Matrix, data[:, s]), data[:, y_idx], regtype, get_nfsic(CuArray(convert(Matrix{Float64}, data[:, s])), (CuArray(glrmX[i, :])))) for i=1:k]
                else
                    regulariser = [SeparationReg(relative_scale, Float64.(data[:, s]), data[:, y_idx], regtype, get_nfsic((CuArray(Float64.(data[:, s]))), Float32.(CuArray(glrmX[i, :])))) for i=1:k]
                end
            else
                regulariser = SeparationReg(relative_scale, data[:, s], data[:, y_idx], regtype)
            end
        elseif test_reg == "sufficiency"
            separator = params["is_target_feature_categorical"] ? encode_to_one_hot(data[:, y_idx]) : data[:, y_idx]
            if fairness == "hsic"
                regulariser = [SufficiencyReg(relative_scale, data[:, s], separator, regtype, get_nfsic(CuArray(data[:, s]), CuArray(glrmX[i, :]))) for i=1:k]
            else
                regulariser = SufficiencyReg(relative_scale, data[:, s], separator, regtype)
            end
        else
            error("Regulariser $test_reg not implemented yet!")
            regulariser = nothing
        end

        # X_init = copy(glrmX)
        # Y_init = copy(glrmY)

        if args["gpu"]
            X_init = CuArray(glrmX)
            Y_init = CuArray(glrmY)
        else
            X_init = copy(glrmX)
            Y_init = copy(glrmY)
        end
            
        alpha = test_reg == "nothing" ? relative_scale : 1000.0
        if d == "adult"
            fglrm = FairGLRM(data, losses, ZeroReg(), ZeroReg(), regulariser, ZeroColReg(), k, s,
                WeightedLogSumExponentialLoss(alpha, weights),
                X=X_init, Y=Y_init, Z=groups)
        elseif startswith(d, "ad_observatory")
            indices = [(i, j) for i=1:size(data, 1), j=1:size(data, 2)]
            obs = filter(x -> !ismissing(data[x[1], x[2]]), indices)
            fglrm = FairGLRM(data, losses, ZeroReg(), ZeroReg(), regulariser, ZeroColReg(), k, s,
                WeightedLogSumExponentialLoss(alpha, weights),
                X=X_init, Y=Y_init, Z=groups, obs=obs)
        end

        ch = ConvergenceHistory("FairGLRM-$(d)-$(test_reg)-$(fairness)-$(relative_scale)")    
        fglrmX, fglrmY, fair_ch = fit!(fglrm, params=p, ch=ch, verbose=true, checkpoint=true)
        reconstructed = fglrmX' * fglrmY
        fname = "projected_data.csv"
        clustername = cluster > 0 ? "cluster_$(cluster)" : "no_interests"
        dir = "data/results/$d/$(clustername)/$(args["k"])_components/$test_reg/$(fairness)/scale_$scale"
        mkpath(dir)
        fpath = joinpath(dir, fname)
            
        println("successfully fit fair GLRM")
        println("Final loss for this fair GLRM is $(fair_ch.objective[end])")
        CSV.write(fpath, Tables.table(reconstructed))
            
        # if fairness == "hsic"
        #     if typeof(params["protected_characteristic_idx"]) <: Array
        #         hsic_reg = HSICReg(relative_scale, Float64.(convert(Matrix, data[:, s])), glrmX[1, :], HSIC)
        #     else
        #         hsic_reg = HSICReg(relative_scale, Float64.(data[:, s]), glrmX[1, :], HSIC)
        #     end
        #     fair_total_orthog = sum(evaluate(hsic_reg, fglrmX[i, :]) for i=1:k) / relative_scale
        # else
        #     fair_total_orthog = sum(evaluate(fglrm.rkx[i], fglrmX[i, :]) for i=1:k) / scale
        # end
        # println("Penalty for fair GLRM (without scaling) is $fair_total_orthog")
        # println("Penalty for fair GLRM (with scaling) is $(fair_total_orthog * scale)")
    
        fname = "penalty.txt"
        fpath = joinpath(dir, fname)
    
        open(fpath, "w") do file
            # penalty = "Fairness penalty (unscaled): $fair_total_orthog\nFairness penalty (scaled): $(fair_total_orthog * relative_scale)"
            penalty = ""
            write(file, "Loss: $(fair_ch.objective[end])\n$penalty")
        end

        if fairness == "orthog" break end
    end

    println("Finished test for $test_reg using $fairness on the $d dataset at date/time $(now())")
    println()

end