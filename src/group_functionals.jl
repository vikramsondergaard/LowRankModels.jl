import Base: *, convert
import Optim: optimize, LBFGS

export GroupFunctional, WeightedGroupFunctional, UnweightedGroupFunctional,
       StandardGroupLoss, MinMaxLoss, WeightedLPNormLoss, PenalisedLearningLoss,
       WeightedLogSumExponentialLoss, evaluate, grad, grad_x, grad_y, z

abstract type GroupFunctional end                               # The GroupFunctional type
abstract type WeightedGroupFunctional<:GroupFunctional end      # GroupFunctional instances including weight vectors
abstract type UnweightedGroupFunctional<:GroupFunctional end    # GroupFunctional instances without weight vectors

"""
Ensures that the given weighted group functional has a weight vector w that
satisfies the following requirements:

1. For every element wₖ of w, wₖ > 0.
2. The sum of every element of w is equal to 1.

# Parameters
 - `l::WeightedGroupFunctional`: The weighted group functional. Weighted group functionals necessarily
                                 contain weights, hence the typing.
"""
function validate_weights(l::WeightedGroupFunctional)
    # Check criterion 2
    c = isapprox(sum(l.weights), 1.0, atol=0.01)
    @assert isapprox(sum(l.weights), 1.0, atol=0.01) "Expected weights to sum to 1.0, but they instead summed to $(sum(l.weights))"
    # Check criterion 1
    for w in l.weights @assert w >= 0 end
end

"""
Computes zₖ as per the definition in "Towards Fair Unsupervised Learning".

# Parameters
- `losses::Array{Loss, 1}`: The column-wise losses of the fair GLRM.
- `XY`:                     The current best low-rank approximation of A.
- `A`:                      The given data matrix.
- `magnitude_Ωₖ::Float64`:  The number of entries belonging to group k of the
                            protected characteristic.

# Returns
The value of zₖ.
"""
function z(losses::Array{<:Loss, 1}, XY, A, magnitude_Ωₖ::Int64)
    m,n = size(A)
    # Get the y indices corresponding to each loss function
    yidxs = get_yidxs(losses)
    z = 0.0
    for i=1:m
        # Note that we only care about the features observed by the actual GLRM
        for j=1:n
            # Add each loss to the total loss
            z += evaluate(losses[j], XY[i, yidxs[j]], A[i, j])
        end
    end
    # Normalise the loss value corresponding to the size of the data belonging
    # to the given group
    z / magnitude_Ωₖ
end

"""
Computes zₖ as per the definition in "Towards Fair Unsupervised Learning".

# Parameters
- `loss::L where L<:Loss`: The given loss with which to compute zₖ.
- `u::Real`:               The current best low-rank approximation of A_{ij}.
- `a::Number`:             The actual value of Aᵢⱼ.
- `magnitude_Ωₖ::Int64`:   The number of entries belonging to group k of the
                           protected characteristic.

# Returns
The value of zₖ.
"""
function z(loss::L where L<:Loss, u::Real, a::Number, magnitude_Ωₖ::Int64)
    if isa(loss, MultinomialLoss) || isa(loss, OvALoss)
        u = Int(u)
        eval = evaluate(loss, [u == i for i=1:embedding_dim(loss)], Int(a)) / magnitude_Ωₖ
    elseif isa(loss, WeightedHingeLoss)
        eval = evaluate(loss, u, Bool(a)) / magnitude_Ωₖ
    else
        eval = evaluate(loss, u, a) / magnitude_Ωₖ
    end
    eval
end

"""
Computes zₖ as per the definition in "Towards Fair Unsupervised Learning".

# Parameters
- `loss::L where L<:Loss`: The given loss with which to compute zₖ.
- `u::Vector{Real}`:       The current best low-rank approximation of A_{ij}.
- `a::Number`:             The actual value of Aᵢⱼ.
- `magnitude_Ωₖ::Int64`:   The number of entries belonging to group k of the
                           protected characteristic.

# Returns
The value of zₖ.
"""
function z(loss::L where L<:Loss, u::Vector, a::Number, magnitude_Ωₖ::Int64)
    if isa(loss, MultinomialLoss) || isa(loss, OvALoss)
        eval = evaluate(loss, u, a) / magnitude_Ωₖ
    # This is the case for the single-dimension loss functions.
    else
        eval = sum(evaluate(loss, e, a) for e in u) / magnitude_Ωₖ
    end
    eval
end

"""
Computes the gradient of zₖ. This is required for the gradients of several
group functionals.

# Parameters
- `losses::Array{Loss, 1}`: The column-wise losses of the fair GLRM.
- `XY`:                     The current best low-rank approximation of A.
- `A`:                      The given data matrix.
- `magnitude_Ωₖ::Float64`:  The number of entries belonging to group k of the
                            protected characteristic.

# Returns
The gradient of zₖ.
"""
function grad_z(losses::Array{Loss, 1}, XY, A, magnitude_Ωₖ)
    m,n = size(A)
    # Get the y indices corresponding to each loss function
    yidxs = get_yidxs(losses)
    grad = 0.0
    for i=1:m
        # Note that we only care about the features observed by the actual GLRM
        for j=1:n
            # Add each individual gradient to the total gradient
            grad += grad(losses[j], XY[i, yidxs[j]], A[i, j])
        end
    end
    # Normalise the gradient value corresponding to the size of the data
    # belonging to the given group
    grad / magnitude_Ωₖ
end

######################## EXAMPLES #########################

mutable struct StandardLoss<:WeightedGroupFunctional
    weights::Array{Float64}
    Z::Array{Float64}
    magnitude_Ωₖ::Array{Int64}
end

function evaluate(l::StandardLoss, losses::Array{<:Loss, 1}, XY,
        A, Z, observed_features;
        yidxs = get_yidxs(losses))
    # Need to validate the weights are non-negative and normalised
    validate_weights(l)
    # This is the weighted exponential part of the output
    ∑ₖwₖzₖ = 0.0
    one_dim_XY = length(size(XY)) == 1
    one_dim_A = length(size(A)) == 1
    for (k, group) in enumerate(Z)
        z_k = 0.0
        # Get the magnitude of Ωₖ for calculating zₖ
        for i in group
            for j in observed_features[i]
                # Calculate zₖ per row in Ωₖ
                lossj = losses[j]
                yidxsj = yidxs[j]
                if one_dim_XY XYij = XY[yidxsj] else XYij = XY[i, yidxsj] end
                if one_dim_A  Aij  = A[j]       else Aij  = A[i, j]       end
                z_k += z(lossj, XYij, Aij, l.magnitude_Ωₖ[k])
            end
        end
        ∑ₖwₖzₖ += l.weights[k] * z_k
    end
    ∑ₖwₖzₖ
end

"""
The weighted log-sum-exponential group loss. This computes
log(∑ₖ wₖeᵅᶻ⁽ᵏ⁾) / α for k ∈ [1, K].

# Parameters
- `α::Float64`:              A coefficient which determines how much debiasing
                             should be performed, at the expense of accuracy.
                             Higher values of α correspond to prioritising
                             fairness more over accuracy. In their experiments,
                             Buet-Golfouse and Utyagulov use α ∈ [10⁻⁶, 10⁵].
- `weights::Array{Float64}`: The weights with which to compute the weighted sum
                             of each individual loss.
"""
mutable struct WeightedLogSumExponentialLoss<:WeightedGroupFunctional
    α::Float64
    weights::Array{Float64}
    Z::Array{Float64}
    magnitude_Ωₖ::Array{Int64}
end

WeightedLogSumExponentialLoss(α::Float64, weights::Array{Float64}) =
    WeightedLogSumExponentialLoss(α, weights, zeros(Float64, length(weights)), zeros(Int64, length(weights)))

"""
Evaluates the total loss for given data. Note that this method is often adapted
with smaller datasets: see the `row_objective()` and `col_objective()`
functions for fair GLRMs in `evaluate_fit.jl` for an example of such use. Note
that this function does not provide any help with targeting specific areas of
the data. It is general enough that it can handle most data, if well-specified.

# Parameters
- `l::WeightedLogSumExponentialLoss`: The weighted log-sum-exponential group
                                      loss instance whose weights and α value
                                      are key to computing this loss.
- `losses::Array{<:Loss, 1}`:         The column-wise losses of the fair GLRM.
- `XY`:                               The current best low-rank approximation
                                      of A.
- `A`:                                The given data matrix.
- `Z`:                                The partitioning of the data by protected
                                      characteristic value.
- `observed_features`:                The observed features for each row of the
                                      data.
- `yidxs`:                            The mapping of column indices from A to
                                      XY. More often than not this will just be
                                      the identity.

# Returns
The groupwise loss as per the definition of the weighted log-sum-exponential
loss.
"""
function evaluate(l::WeightedLogSumExponentialLoss, losses::Array{<:Loss, 1}, XY,
                  A, Z, observed_features;
                  yidxs = get_yidxs(losses))
    # α needs to be greater than 0 because it is the denominator of our end
    # result
    @assert l.α > 0
    # Need to validate the weights are non-negative and normalised
    validate_weights(l)
    # This is the weighted exponential part of the output
    ∑ₖwₖeᵅᶻ = 0.0
    one_dim_XY = length(size(XY)) == 1
    one_dim_A = length(size(A)) == 1
    for (k, group) in enumerate(Z)
        z_k = 0.0
        # Get the magnitude of Ωₖ for calculating zₖ
        size_Ωₖ = sum(length(observed_features[i]) for i in group)
        for i in group
            for j in observed_features[i]
                # Calculate zₖ per row in Ωₖ
                lossj = losses[j]
                yidxsj = yidxs[j]
                if one_dim_XY XYij = XY[yidxsj] else XYij = XY[i, yidxsj] end
                if one_dim_A  Aij  = A[j]       else Aij  = A[i, j]       end
                z_k += z(lossj, XYij, Aij, size_Ωₖ)
            end
        end
        # Now that we have zₖ, calculate the exponential part of the weighted
        # log-sum exponential
        eᵅᶻᵏ = exp(l.α * z_k)
        # println("eᵅᶻᵏ is: $eᵅᶻᵏ for group $k")
        # Multiply by the corresponding weight to get the weighted part of the
        # weighted log-sum exponential
        ∑ₖwₖeᵅᶻ += l.weights[k] * eᵅᶻᵏ
    end
    # Divide by α as per the specification
    log(∑ₖwₖeᵅᶻ) / l.α
end

"""
Evaluates the gradient of the data at a given point. Unlike `evaluate()`, this
function is not over the whole data, rather it is over just one coordinate
(i, j). This choice was made because the existing prox-grad method evaluated
the gradient of each coordinate separately. Realising that this could also be
done for the weighted log-sum-exponential group loss meant I coded it this way
too.

Note that, in theory, `evaluate()` can also be used point-wise: it just takes a
lot more effort to set up the parameters the right way to do this.

# Parameters
- `l::WeightedLogSumExponentialLoss`: The weighted log-sum-exponential group
                                      loss instance whose weights and α value
                                      are key to computing this loss.
- `i`:                                The row of the coordinate whose gradient
                                      is being computed.
- `j`:                                The column of the coordinate whose
                                      gradient is being computed.
- `losses::Array{Loss, 1}`:           The column-wise losses of the fair GLRM.
- `XY`:                               The current best low-rank approximation
                                      of A.
- `A`:                                The given data matrix.
- `Z`:                                The partitioning of the data by protected
                                      characteristic value.
- `observed_features`:                The observed features for each row of the
                                      data.
- `yidxs`:                            The mapping of column indices from A to
                                      XY. More often than not this will just be
                                      the identity.
"""
function grad(l::WeightedLogSumExponentialLoss, i, j, losses::Array{Loss, 1},
                XY, A, Z, observed_features; 
                yidxs = get_yidxs(losses), refresh = (i * j == 1))
    # Because the group-wise losses only update once per gradient step,
    # it makes sense to cache these and refresh only when needed
    if refresh
        for (k, group) in enumerate(Z)
            z_k = 0.0 # The loss for group k ∈ [1, K]
            size_Ωₖ = sum(length(observed_features[g]) for g in group)
            for g in group
                for f in observed_features[g]
                    # Add the loss for each row in the k-th group
                    z_k += z(losses[f], XY[g, yidxs[f]], A[g, f], size_Ωₖ)
                end
            end
            # Compute the exponential as defined by Buet-Golfouse and Utyagulov
            eᵅᶻᵏ = exp(l.α * z_k)
            # Cache this in the WSE loss instance so we don't have to compute
            # it again later
            l.Z[k] = eᵅᶻᵏ
            l.magnitude_Ωₖ[k] = size_Ωₖ
        end
    end
    # This is the group that row i belongs in
    k_i = 0
    for (k, group) in enumerate(Z)
        if (i in group) k_i = k; break end
    end
    # Get the product of the k-th weight and the k-th exponential
    wₖ₍ᵢ₎eᵅᶻ = l.weights[k_i] * l.Z[k_i]
    ∑ₖwₖeᵅᶻ = sum(l.weights .* l.Z)
    loss = losses[j]
    # grad(losses[j], XY[i, yidxs[j]], A[i, j]) * wₖ₍ᵢ₎eᵅᶻ / (size_Ωₖ₍ᵢ₎ * ∑ₖwₖeᵅᶻ)
    u = XY[i, yidxs[j]]
    a = A[i, j]
    
    if isa(loss, MultinomialLoss) || isa(loss, OvALoss)
        if isa(u, Number)
            u = Int(u)
            g = grad(loss, [u == i for i=1:embedding_dim(loss)], Int(a))
        else
            g = grad(loss, u, a)
        end
    elseif isa(loss, WeightedHingeLoss)
        g = grad(loss, u, Bool(a))
    else
        g = grad(loss, u, a)
    end
    g * wₖ₍ᵢ₎eᵅᶻ / (l.magnitude_Ωₖ[k_i] * ∑ₖwₖeᵅᶻ)
end