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
    for w in l.weights
        @assert w >= 0
    end
end

"""
Computes zₖ as per the definition in "Towards Fair Unsupervised Learning".

# Parameters
- `fglrm::FairGLRM`: The Fair GLRM. This is needed to get information about the
                     division of groups among the data, the observed features
                     in the Fair GLRM, and the loss functions corresponding to
                     each column of data.
- `XY`:              The computed matrix product of X and Y.
- `k`:               The given group for which zₖ will be computed.

# Returns
The value of zₖ.
"""
function z(losses::Array{Loss, 1}, XY, A, magnitude_Ωₖ::Float64)
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

z(loss, u, a, magnitude_Ωₖ) = evaluate(loss, u, a) / magnitude_Ωₖ

"""
Computes the gradient of zₖ. This is useful for several gradients.

# Parameters
- `fglrm::FairGLRM`: The Fair GLRM. This is needed to get information about the
                     division of groups among the data, the observed features
                     in the Fair GLRM, and the loss functions corresponding to
                     each column of data.
- `XY`:              The computed matrix product of X and Y.
- `k`:               The given group for which the gradient of zₖ will be 
                     computed.

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

mutable struct WeightedLogSumExponentialLoss<:WeightedGroupFunctional
    α::Float64
    weights::Array{Float64}
end

function evaluate(l::WeightedLogSumExponentialLoss, losses::Array{Loss, 1}, XY, A, Z, observed_features; yidxs = get_yidxs(losses))
    @assert l.α > 0
    validate_weights(l)
    m,n = size(A)
    err = 0.0
    for i=1:m
        k = 0
        for (e, group) in enumerate(Z)
            if i in group k = e; break end
        end
        for j in observed_features[i]
            z_ij = z(losses[j], XY[i, yidxs[j]], A[i, j], length(Z[k]))
            err += exp(l.α * z_ij) * l.weights[k]
        end
    end
    log(err) / l.α
end

# function grad(l::WeightedLogSumExponentialLoss, losses::Array{Loss, 1}, XY, A, Z; yidxs = get_yidxs(losses))
#     gradient = sum([l.weights[k] * grad_z(losses, XY, A, k) * exp(l.α * z(fglrm, XY, k)) for k=1:size(Z)[1]])
#     denom = sum([l.weights[k] * exp(l.α * z(fglrm, XY, k)) for k=1:size(fglrm.Z)[1]])
#     gradient / denom
# end

function grad_x(l::WeightedLogSumExponentialLoss, i, j, losses::Array{Loss, 1},
                XY, A, Z, observed_features; 
                yidxs = get_yidxs(losses))
    k_i = 0
    for (k, group) in enumerate(Z)
        if (i in group) (k_i = k; break) end
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

# function grad_x(l::WeightedLogSumExponentialLoss, i, j, losses::Array{Loss, 1},
#                 XY, A, Z, observed_features; 
#                 yidxs = get_yidxs(losses))
#     m,n = size(A)
#     k_i = 0
#     for (k, group) in enumerate(Z)
#         if i in group
#             k_i = k
#             break
#         end
#     end
#     size_Ωₖ₍ᵢ₎ = length(Z[k_i])
#     exp_ki = exp(l.α * z(losses, XY[i, yidxs], A[i, yidxs], size_Ωₖ₍ᵢ₎))
#     dT_dz_ki_n = l.weights[k_i] * exp_ki
#     dT_dz_ki_d = 0.0
#     for (k, group) in enumerate(Z)
#         size_Ωₖ = length(group)
#         exp_k = exp(l.α * z(losses, XY[group, yidxs], A[group, yidxs], size_Ωₖ))
#         dT_dz_ki_d += l.weights[k] * exp_k
#     end
#     dT_dz_ki = dT_dz_ki_n / dT_dz_ki_d
#     dz_ki_dx_i = sum([grad(losses[j], XY[i, yidxs[j]], A[i, j]) * XY[i, yidxs[j]] for j=1:n])
#     dT_dz_ki * dz_ki_dx_i / size_Ωₖ₍ᵢ₎
# end

# function grad_x(l::WeightedLogSumExponentialLoss, loss::Loss, u::Real, a::Number, k::Int64, size_Ωₖ::Int64, XY, A, Z)
#     w_k = l.weights[k]
#     e = exp(l.α * z(losses, )))
#     denom = sum([l.weights[k] * exp(l.α * )])
#     return grad(loss, u, a) * w_k * e / (l.denom * size_Ωₖ)
# end

function grad_y(l::WeightedLogSumExponentialLoss, fglrm::FairGLRM, XY, j)
    gradient = 0.0
    normalised_den = sum([l.weights[k] * exp(l.α * z(fglrm, XY, k)) for k=1:size(fglrm.Z)[1]])
    loss = fglrm.losses[j]
    yidx = get_yidxs(fglrm.losses)[j]
    for k=1:size(fglrm.Z)[1]
        dT_dz_k = l.weights[k] * exp(l.α * z(fglrm, XY, k)) / normalised_den
        magnitude_Ωₖ = length(fglrm.Z[k])
        dz_k_dy_j = sum([grad(loss, XY[i, yidx], fglrm.A[i, j]) * XY[i, yidx] for i in fglrm.Z[k]])
        gradient += dT_dz_k * dz_k_dy_j / magnitude_Ωₖ
    end
    gradient
end