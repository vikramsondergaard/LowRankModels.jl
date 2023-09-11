import Base: *, convert
import Optim: optimize, LBFGS

export GroupFunctional, WeightedGroupFunctional, UnweightedGroupFunctional,
       StandardGroupLoss, MinMaxLoss, WeightedLPNormLoss, PenalisedLearningLoss,
       WeightedLogSumExponentialLoss, evaluate, grad, grad_x, grad_y

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
    @assert isapprox(sum(l.weights), 1.0, atol=0.01)
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
    # This is the case where the loss function is multi-dimensional. Need to
    # convert `u` to a singleton list.
    if !isa(loss, SingleDimLoss) && !isa(loss, OrdinalHingeLoss)
        eval = evaluate(loss, [u], a) / magnitude_Ωₖ
    # This is the case for the single-dimension loss functions.
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
function z(loss::L where L<:Loss, u::Vector{Float64}, a::Number, magnitude_Ωₖ::Int64)
    if !isa(loss, SingleDimLoss) && !isa(loss, OrdinalHingeLoss)
        eval = evaluate(loss, u, a) / magnitude_Ωₖ
    # This is the case for the single-dimension loss functions.
    else
        eval = sum(evaluate(loss, e, a) for e in u) / magnitude_Ωₖ
    end
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

"""
The standard group loss. This computes ∑ₖ wₖzₖ for k ∈ [1, K].

# Parameters
- `weights::Array{Float64}`: The weights with which to compute the weighted sum
                             of each individual loss.
"""
# mutable struct StandardGroupLoss<:WeightedGroupFunctional
#     weights::Array{Float64}      # The weights applied to each element of the group
# end

# """
# Evaluates the standard group loss.

# # Parameters
# - `l::StandardGroupLoss`: The given loss instance.
# - `fglrm::FairGLRM`:      The Fair GLRM. This is needed to get information about the
#                           division of groups among the data, the observed features
#                           in the Fair GLRM, and the loss functions corresponding to
#                           each column of data.
# - `XY`:                   The computed matrix product of X and Y.

# # Returns
# The total error for a given array of groupwise losses according to the
# StandardGroupLoss rules.
# """
# function evaluate(l::StandardGroupLoss, fglrm::FairGLRM, u::Real, a::Number, i::Int64, j::Int64)
#     # Need to validate the weights vector because it is a weighted group loss
#     validate_weights(l)
#     k = find_group(fglrm, i)
#     evaluate(l, fglrm, u, a, i, j, k)
# end

# function evaluate(l::StandardGroupLoss, loss, u::Real, a::Number, i::Int64, j::Int64, k::Int64; val_weights::Bool=false)
#     if val_weights validate_weights(l) end
#     l.weights[k] * z(get_yidxs(fglrm.losses)[j], u, a, length(fglrm.Z[k]))
# end

# """
# Computes the gradient of the standard group loss.

# # Parameters
# - `l::StandardGroupLoss`: The given loss instance.
# - `fglrm::FairGLRM`:      The Fair GLRM. This is needed to get information about the
#                           division of groups among the data, the observed features
#                           in the Fair GLRM, and the loss functions corresponding to
#                           each column of data.
# - `XY`:                   The computed matrix product of X and Y.

# # Returns
# The gradient for a given array of groupwise losses according to the
# standard group loss rules.
# """
# grad(l::StandardGroupLoss, fglrm::FairGLRM, XY) =
#  sum([l.weights[k] * grad_z(fglrm, XY, k) for k=1:size(fglrm.Z)[1]]) # ∑ₖ wₖ∇zₖ for k ∈ [1,K]

# """
# The MinMax group loss. This computes maxₖ(zₖ) for k ∈ [1, K].
# """
# mutable struct MinMaxLoss<:UnweightedGroupFunctional end

# """
# Evaluates the MinMax group loss.

# # Parameters
# - `l::MinMaxLoss`:   The given loss instance.
# - `fglrm::FairGLRM`: The Fair GLRM. This is needed to get information about the
#                      division of groups among the data, the observed features
#                      in the Fair GLRM, and the loss functions corresponding to
#                      each column of data.
# - `XY`:              The computed matrix product of X and Y.

# # Returns
# The total error for a given array of groupwise losses according to the
# MinMax group loss rules.
# """
# evaluate(l::MinMaxLoss, fglrm::FairGLRM, XY) = max([z(fglrm, XY, k) for k=1:size(fglrm.Z)[1]]) # maxₖ(zₖ) for k ∈ [1, K]

# """
# The weighted Lᵖ group loss. This computes (∑ₖ wₖzₖᵖ)^(1/p) for k ∈ [1, K].

# # Parameters
# - `weights::Array{Float64}`: The weights with which to compute the weighted sum
#                              of each individual loss.
# - `p::Float64`:              The type of norm to compute.
# """
# mutable struct WeightedLᵖNormLoss<:WeightedGroupFunctional
#     weights::Array{Float64}
#     p::Float64
# end

# """
# Evaluates the weighted Lᵖ group loss.

# # Parameters
# - `l::WeightedLᵖNormLoss`: The given loss instance.
# - `fglrm::FairGLRM`:       The Fair GLRM. This is needed to get information about the
#                            division of groups among the data, the observed features
#                            in the Fair GLRM, and the loss functions corresponding to
#                            each column of data.
# - `XY`:                    The computed matrix product of X and Y.
 
# # Returns
# The total error for a given array of groupwise losses according to the
# weighted Lᵖ group loss rules.
# """
# function evaluate(l::WeightedLᵖNormLoss, fglrm::FairGLRM, XY)
#     # Need to validate the weights vector because it is a weighted group loss
#     validate_weights(l)
#     # (∑ₖ wₖzₖᵖ)^(1/p) for k ∈ [1, K]
#     total_err = sum([l.weights[k] * z(fglrm, XY, k)^l.p for k=1:size(fglrm.Z)[1]])^(1.0/l.p)
#     total_err
# end

# """
# Computes the gradient of the weighted Lᵖ norm group loss.

# # Parameters
# - `l::WeightedLᵖNormLoss`: The given loss instance.
# - `fglrm::FairGLRM`:       The Fair GLRM. This is needed to get information about the
#                            division of groups among the data, the observed features
#                            in the Fair GLRM, and the loss functions corresponding to
#                            each column of data.
# - `XY`:                    The computed matrix product of X and Y.

# # Returns
# The gradient for a given array of groupwise losses according to the
# weighted Lᵖ norm group loss rules.
# """
# function grad(l::WeightedLᵖNormLoss, fglrm::FairGLRM, XY)
#     # This is computed from the chain product rule
#     inner_sum = sum([l.weights[k] * z(fglrm, XY, k)^(p-1) for k=1:size(fglrm.Z)[1]])
#     outer_sum = sum([l.weights[k] * z(fglrm, XY, k)^(p) for k=1:size(fglrm.Z)[1]])^(1/p - 1)
#     # This is computed from the chain product of the inner sum
#     gradient = sum([grad_z(fglrm, XY, k) for k=1:size(fglrm.Z)[1]])
#     inner_sum * outer_sum * gradient
# end

# """
# The penalised learning group loss. This computes ∑ₖ wₖzₖ + λ∑ₖ₁ₖ₂ d(zₖ₁, zₖ₂) for k, k₁, k₂ ∈ [1, K].

# # Parameters
# - `weights::Array{Float64}`: The weights with which to compute the weighted sum
#                              of each individual loss.
# - `λ::Float64`:              The trade-off between the loss and the distance penalty.
# - `d::Function`:             The chosen distance penalty.
# - `d_grad::Function`:        The gradient of the chosen distance
# """
# mutable struct PenalisedLearningLoss<:WeightedGroupFunctional
#     weights::Array{Float64}
#     λ::Float64
#     d::Function
#     d_grad::Function
# end

# """
# # Parameters
# - `l::PenalisedLearningLoss`: The given loss instance.
# - `groups::Array{Float64}`:   The given groupwise losses.

# # Returns
# The total error for a given array of groupwise losses according to the
# PenalisedLearningLoss rules.
# """
# function evaluate(l::PenalisedLearningLoss, fglrm::FairGLRM, XY)
#     validate_weights(l)
#     @assert l.λ > 0
#     err = sum([l.weights[k] * z(fglrm, XY, k) for k=1:size(fglrm.Z)[1]])
#     penalty = l.λ * sum([d(z(fglrm, XY, k), z(fglrm, XY, k_prime)) for k=1:size(fglrm.Z)[1], k_prime=1:size(fglrm.Z)[1]])
#     err + penalty
# end

# function grad(l::PenalisedLearningLoss, fglrm::FairGLRM, XY)
#     gradient = sum([l.weights[k] * grad_z(fglrm, XY, k) for k=1:size(fglrm.Z)[1]])
#     for k=1:size(fglrm.Z)[1]
#         for k_prime=1:size(fglrm.Z)[1]
#             gradient += l.λ * d_grad(z(fglrm, XY, k), z(fglrm, XY, k_prime))
#         end
#     end
#     gradient
# end

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
end

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
    print("yidxs: "); display(yidxs)
    print("XY: ");    display(XY)
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
        size_Ωₖ = length(group)
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
                yidxs = get_yidxs(losses))
    # TODO: comments!!
    k_i = 0
    for (k, group) in enumerate(Z)
        if (i in group) k_i = k; break end
    end
    size_Ωₖ₍ᵢ₎ = length(Z[k_i])
    z_ki = 0.0
    for i in Z[k_i]
        for j in observed_features[i]
            z_ki += z(losses[j], XY[i, yidxs[j]], A[i, j], size_Ωₖ₍ᵢ₎)
        end
    end
    eᵅᶻ = exp(l.α * z_ki)
    wₖeᵅᶻ = l.weights[k_i] * eᵅᶻ
    ∑ₖwₖeᵅᶻ = 0.0
    for (k, group) in enumerate(Z)
        z_k = 0.0
        size_Ωₖ = length(group)
        for i in group
            for j in observed_features[i]
                z_k += z(losses[j], XY[i, yidxs[j]], A[i, j], size_Ωₖ)
            end
        end
        eᵅᶻᵏ = exp(l.α * z_k)
        ∑ₖwₖeᵅᶻ += l.weights[k] * eᵅᶻᵏ
    end
    grad(losses[j], XY[i, yidxs[j]], A[i, j]) * wₖeᵅᶻ / (size_Ωₖ₍ᵢ₎ * ∑ₖwₖeᵅᶻ)
end