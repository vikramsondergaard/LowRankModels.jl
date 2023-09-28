### Proximal gradient method
export GradParams, fit!

mutable struct GradParams<:AbstractParams
    stepsize::Float64 # initial stepsize
    max_iter::Int # maximum number of outer iterations
    inner_iter_X::Int # how many prox grad steps to take on X before moving on to Y (and vice versa)
    inner_iter_Y::Int # how many prox grad steps to take on Y before moving on to X (and vice versa)
    abs_tol::Float64 # stop if objective decrease upon one outer iteration is less than this * number of observations
    rel_tol::Float64 # stop if objective decrease upon one outer iteration is less than this * objective value
    min_stepsize::Float64 # use a decreasing stepsize, stop when reaches min_stepsize
end
function GradParams(stepsize::Number=1.0; # initial stepsize
				        max_iter::Int=100, # maximum number of outer iterations
                inner_iter_X::Int=1, # how many prox grad steps to take on X before moving on to Y (and vice versa)
                inner_iter_Y::Int=1, # how many prox grad steps to take on Y before moving on to X (and vice versa)
                inner_iter::Int=1,
                abs_tol::Number=0.00001, # stop if objective decrease upon one outer iteration is less than this * number of observations
                rel_tol::Number=0.0001, # stop if objective decrease upon one outer iteration is less than this * objective value
				        min_stepsize::Number=0.01*stepsize) # stop if stepsize gets this small
    stepsize = convert(Float64, stepsize)
    inner_iter_X = max(inner_iter_X, inner_iter)
    inner_iter_Y = max(inner_iter_Y, inner_iter)
    return GradParams(convert(Float64, stepsize),
                          max_iter,
                          inner_iter_X,
                          inner_iter_Y,
                          convert(Float64, abs_tol),
                          convert(Float64, rel_tol),
                          convert(Float64, min_stepsize))
end

function fit!(glrm::GLRM, params::GradParams;
              ch::ConvergenceHistory=ConvergenceHistory("GradGLRM"),
              verbose=true,
              kwargs...)
    ### initialization
	A = glrm.A # rename these for easier local access
	losses = glrm.losses
	rx = glrm.rx
	ry = glrm.ry
	X = glrm.X; Y = glrm.Y
    # check that we didn't initialize to zero (otherwise we will never move)
    if norm(Y) == 0
    	Y = .1*randn(k,d)
    end
	k = glrm.k
    m,n = size(A)

    # find spans of loss functions (for multidimensional losses)
    yidxs = get_yidxs(losses)
    d = maximum(yidxs[end])
    # check Y is the right size
    if d != size(Y,2)
        @warn("The width of Y should match the embedding dimension of the losses.
            Instead, embedding_dim(glrm.losses) = $(embedding_dim(glrm.losses))
            and size(glrm.Y, 2) = $(size(glrm.Y, 2)).
            Reinitializing Y as randn(glrm.k, embedding_dim(glrm.losses).")
            # Please modify Y or the embedding dimension of the losses to match,
            # eg, by setting `glrm.Y = randn(glrm.k, embedding_dim(glrm.losses))`")
        glrm.Y = randn(glrm.k, d)
    end

    XY = Array{Float64}(undef, (m, d))
    gemm!('T','N',1.0,X,Y,0.0,XY) # XY = X' * Y initial calculation

    # alternating updates of X and Y
    if verbose println("Fitting GLRM") end
    obj = objective(glrm, X, Y, XY, yidxs=yidxs)
    update_ch!(ch, 0, obj)
    t = time()
    steps_in_a_row = 0
    # gradient wrt columns of X
    g = zeros(k)
    # gradient wrt column-chunks of Y
    G = zeros(k, d)

    scaled_abs_tol = params.abs_tol * mapreduce(length,+,glrm.observed_features)

    # cache views for better memory management
    # make sure we don't try to access memory not allocated to us
    @assert(size(Y) == (k,d))
    @assert(size(X) == (k,m))
    # views of the columns of X corresponding to each example
    ve = [view(X,:,e) for e=1:m]
    # views of the column-chunks of Y corresponding to each feature y_j
    # vf[f] == Y[:,f]
    vf = [view(Y,:,yidxs[f]) for f=1:n]
    # views of the column-chunks of G corresponding to the gradient wrt each feature y_j
    # these have the same shape as y_j
    gf = [view(G,:,yidxs[f]) for f=1:n]

    # working variables
    newX = copy(X)
    newY = copy(Y)
    newve = [view(newX,:,e) for e=1:m]
    newvf = [view(newY,:,yidxs[f]) for f=1:n]

    for i=1:params.max_iter

        stepsize = 0.5 / i
        println("The value of stepsize at timestep $i is $stepsize")

        for e=1:m # for every example x_e == ve[e]
            fill!(g, 0.) # reset gradient to 0
            # compute gradient of L with respect to Xᵢ as follows:
            # ∇{Xᵢ}L = Σⱼ dLⱼ(XᵢYⱼ)/dXᵢ
            for f in glrm.observed_features[e]
                # but we have no function dLⱼ/dXᵢ, only dLⱼ/d(XᵢYⱼ) aka dLⱼ/du
                # by chain rule, the result is: Σⱼ (dLⱼ(XᵢYⱼ)/du * Yⱼ), where dLⱼ/du is our grad() function
                curgrad = grad(losses[f],XY[e,yidxs[f]],A[e,f])
                if i == 1
                    println("For cell ($e, $f), gradient wrt X for GLRM is $curgrad")
                end
                if isa(curgrad, Number)
                    axpy!(curgrad, vf[f], g)
                else
                    # on v0.4: gemm!('N', 'T', 1.0, vf[f], curgrad, 1.0, g)
                    gemm!('N', 'N', 1.0, vf[f], curgrad, 1.0, g)
                end
            end
            # newx = prox(rx[e], ve[e] - stepsize*g, stepsize) # this will use much more memory than the inplace version with linesearch below
            ## gradient step: Xᵢ += -(α/l) * ∇{Xᵢ}L
            axpy!(-stepsize,g,newve[e])
            ## prox step: Xᵢ = prox_rx(Xᵢ, α/l)
            prox!(rx[e],newve[e],stepsize)
            copyto!(ve[e], newve[e])
        end
        gemm!('T','N',1.0,X,Y,0.0,XY) # Recalculate XY using the new X

        fill!(G, 0.)
        for f=1:n
            for e in glrm.observed_examples[f]
                # but we have no function dLⱼ/dYⱼ, only dLⱼ/d(XᵢYⱼ) aka dLⱼ/du
                # by chain rule, the result is: Σⱼ dLⱼ(XᵢYⱼ)/du * Xᵢ, where dLⱼ/du is our grad() function
                curgrad = grad(losses[f],XY[e,yidxs[f]],A[e,f])
                if i == 1
                    println("For cell ($e, $f), gradient wrt Y for GLRM is $curgrad")
                end
                if isa(curgrad, Number)
                    axpy!(curgrad, ve[e], gf[f])
                else
                    # on v0.4: gemm!('N', 'T', 1.0, ve[e], curgrad, 1.0, gf[f])
                    gemm!('N', 'T', 1.0, ve[e], curgrad, 1.0, gf[f])
                end
            end
            # newy = prox(ry[f], vf[f] - stepsize*gf[f], stepsize)
            ## gradient step: Yⱼ += -(α/l) * ∇{Yⱼ}L
            axpy!(-stepsize,gf[f],newvf[f])
            ## prox step: Yⱼ = prox_ryⱼ(Yⱼ, α/l)
            prox!(ry[f],newvf[f],stepsize)
            copyto!(vf[f], newvf[f])
        end
        gemm!('T','N',1.0,X,Y,0.0,XY) # Recalculate XY using the new Y

        obj = objective(glrm, X, Y, XY, yidxs=yidxs)
        t = time() - t
        update_ch!(ch, t, obj)
        t = time()
        # STEP 4: Check stopping criterion
        obj_decrease = ch.objective[end-1] - obj
        if i>10 && (obj_decrease < scaled_abs_tol || obj_decrease/obj < params.rel_tol)
            break
        end
        if verbose
            println("Iteration $i: objective value = $(ch.objective[end])")
        end

    end

    return glrm.X, glrm.Y, ch
end

function fit!(glrm::FairGLRM, params::GradParams;
              ch::ConvergenceHistory=ConvergenceHistory("GradGLRM"),
              verbose=true,
              kwargs...)
    ### initialization
    A = glrm.A # rename these for easier local access
    losses = glrm.losses
    rx = glrm.rx
    ry = glrm.ry
    X = glrm.X; Y = glrm.Y
    # check that we didn't initialize to zero (otherwise we will never move)
    if norm(Y) == 0
        Y = .1*randn(k,d)
    end
    k = glrm.k
    m,n = size(A)

    group_func = glrm.group_functional
    Z = glrm.Z

    # find spans of loss functions (for multidimensional losses)
    yidxs = get_yidxs(losses)
    d = maximum(yidxs[end])
    # check Y is the right size
    if d != size(Y,2)
        @warn("The width of Y should match the embedding dimension of the losses.
        Instead, embedding_dim(glrm.losses) = $(embedding_dim(glrm.losses))
        and size(glrm.Y, 2) = $(size(glrm.Y, 2)).
        Reinitializing Y as randn(glrm.k, embedding_dim(glrm.losses).")
        # Please modify Y or the embedding dimension of the losses to match,
        # eg, by setting `glrm.Y = randn(glrm.k, embedding_dim(glrm.losses))`")
        glrm.Y = randn(glrm.k, d)
    end

    XY = Array{Float64}(undef, (m, d))
    gemm!('T','N',1.0,X,Y,0.0,XY) # XY = X' * Y initial calculation

    # step size (will be scaled below to ensure it never exceeds 1/\|g\|_2 or so for any subproblem)
    alpharow = params.stepsize*ones(m)
    alphacol = params.stepsize*ones(n)
    # stopping criterion: stop when decrease in objective < tol, scaled by the number of observations
    scaled_abs_tol = params.abs_tol * mapreduce(length,+,glrm.observed_features)

    # alternating updates of X and Y
    if verbose println("Fitting GLRM") end
    obj = objective(glrm, X, Y, XY, yidxs=yidxs)
    update_ch!(ch, 0, obj)
    t = time()
    # gradient wrt columns of X
    g = zeros(k)
    # gradient wrt column-chunks of Y
    G = zeros(k, d)

    # cache views for better memory management
    # make sure we don't try to access memory not allocated to us
    @assert(size(Y) == (k,d))
    @assert(size(X) == (k,m))
    # views of the columns of X corresponding to each example
    ve = [view(X,:,e) for e=1:m]
    # views of the column-chunks of Y corresponding to each feature y_j
    # vf[f] == Y[:,f]
    vf = [view(Y,:,yidxs[f]) for f=1:n]
    # views of the column-chunks of G corresponding to the gradient wrt each feature y_j
    # these have the same shape as y_j
    gf = [view(G,:,yidxs[f]) for f=1:n]

    # working variables
    newX = copy(X)
    newY = copy(Y)
    newve = [view(newX,:,e) for e=1:m]
    newvf = [view(newY,:,yidxs[f]) for f=1:n]

    if eltype(glrm.observed_features) == UnitRange{Int64}
        magnitude_Ω = sum(length(f) for f in glrm.observed_features)
    else
        magnitude_Ω = size(glrm.observed_features, 1) * size(glrm.observed_features, 2)
    end

    for i=1:params.max_iter

        stepsize = 0.5 / i
        println("The value of stepsize at timestep $i is $stepsize")

        refresh = true
        for e=1:m # for every example x_e == ve[e]
            fill!(g, 0.) # reset gradient to 0
            # compute gradient of L with respect to Xᵢ as follows:
            # ∇{Xᵢ}L = Σⱼ dLⱼ(XᵢYⱼ)/dXᵢ
            for f in glrm.observed_features[e]
                # but we have no function dLⱼ/dXᵢ, only dLⱼ/d(XᵢYⱼ) aka dLⱼ/du
                # by chain rule, the result is: Σⱼ (dLⱼ(XᵢYⱼ)/du * Yⱼ), where dLⱼ/du is our grad() function
                curgrad = grad(group_func, e, f, losses, XY, A, Z, glrm.observed_features, refresh=refresh)
                curgrad = curgrad * magnitude_Ω
                refresh = false
                if i == 1
                    println("For cell ($e, $f), gradient wrt X for GLRM is $curgrad")
                end
                if isa(curgrad, Number)
                    axpy!(curgrad, vf[f], g)
                else
                    # on v0.4: gemm!('N', 'T', 1.0, vf[f], curgrad, 1.0, g)
                    gemm!('N', 'N', 1.0, vf[f], curgrad, 1.0, g)
                end
            end
            # newx = prox(rx[e], ve[e] - stepsize*g, stepsize) # this will use much more memory than the inplace version with linesearch below
            ## gradient step: Xᵢ += -(α/l) * ∇{Xᵢ}L
            axpy!(-stepsize,g,newve[e])
            ## prox step: Xᵢ = prox_rx(Xᵢ, α/l)
            prox!(rx[e],newve[e],stepsize)
            copyto!(ve[e], newve[e])
        end
        gemm!('T','N',1.0,X,Y,0.0,XY) # Recalculate XY using the new X

        fill!(G, 0.)
        refresh = true
        for f=1:n
            for e in glrm.observed_examples[f]
                # but we have no function dLⱼ/dYⱼ, only dLⱼ/d(XᵢYⱼ) aka dLⱼ/du
                # by chain rule, the result is: Σⱼ dLⱼ(XᵢYⱼ)/du * Xᵢ, where dLⱼ/du is our grad() function
                curgrad = grad(group_func, e, f, losses, XY, A, Z, glrm.observed_features, refresh=refresh)
                curgrad = curgrad * magnitude_Ω
                refresh = false
                if i == 1
                    println("For cell ($e, $f), gradient wrt Y for GLRM is $curgrad")
                end
                if isa(curgrad, Number)
                    axpy!(curgrad, ve[e], gf[f])
                else
                    # on v0.4: gemm!('N', 'T', 1.0, ve[e], curgrad, 1.0, gf[f])
                    gemm!('N', 'T', 1.0, ve[e], curgrad, 1.0, gf[f])
                end
            end
            # newy = prox(ry[f], vf[f] - stepsize*gf[f], stepsize)
            ## gradient step: Yⱼ += -(α/l) * ∇{Yⱼ}L
            axpy!(-stepsize,gf[f],newvf[f])
            ## prox step: Yⱼ = prox_ryⱼ(Yⱼ, α/l)
            prox!(ry[f],newvf[f],stepsize)
            copyto!(vf[f], newvf[f])
        end
        gemm!('T','N',1.0,X,Y,0.0,XY) # Recalculate XY using the new Y

        obj = objective(glrm, X, Y, XY, yidxs=yidxs)
        t = time() - t
        update_ch!(ch, t, obj)
        t = time()
        # STEP 4: Check stopping criterion
        obj_decrease = ch.objective[end-1] - obj
        if i>10 && (obj_decrease < scaled_abs_tol || obj_decrease/obj < params.rel_tol)
            break
        end
        if verbose
            println("Iteration $i: objective value = $(ch.objective[end])")
        end
    end
    return glrm.X, glrm.Y, ch
end