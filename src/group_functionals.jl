import Base: *, convert
import Optim: optimize, LBFGS

export GroupFunctional, WeightedGroupFunctional, UnweightedGroupFunctional,
       StandardGroupLoss, MinMaxLoss, WeightedLPNormLoss, PenalisedLearningLoss,
       evaluate

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
        @assert w > 0
    end
end

######################## EXAMPLES #########################

### STANDARD GROUP LOSS ###
mutable struct StandardGroupLoss<:WeightedGroupFunctional
    k::Int64                     # The number of groups in the data according to a given protected characteristic
    weights::Array{Float64}      # The weights applied to each element of the group
end

"""
A group functional that weights the loss for each group and sums them together.

# Parameters
- `k::Int64`:                The number of groups in the group functional
- `weights::Array{Float64}`: The weights for each groupwise loss. The default value
                             is the uniform distribution, which corresponds to the
                             *reweighed* GLRM. To recover a standard GLRM, one can set
                             `weights` to be wₖ = |Ωₖ|/|Ω|.
"""
StandardGroupLoss(k::Int64; weights::Array{Float64}=fill(1.0/k, k)) = StandardGroupLoss(k, weights)

"""
# Parameters
- `l::StandardGroupLoss`:   The given loss instance.
- `groups::Array{Float64}`: The given groupwise losses.

# Returns
The total error for a given array of groupwise losses according to the
StandardGroupLoss rules.
"""
function evaluate(l::StandardGroupLoss, groups::Array{Float64})
    validate_weights(l)
    total_err = sum([groups[i] * l.weights[i] for i=1:l.k])
    total_err
end

### MINMAX LOSS ###
mutable struct MinMaxLoss<:UnweightedGroupFunctional end
"""
A group functional that returns the maximum loss out of all the groupwise
losses.
"""
MinMaxLoss() = MinMaxLoss()

"""
# Parameters
- `l::MinMaxLoss`:          The given loss instance.
- `groups::Array{Float64}`: The given groupwise losses.

# Returns
The total error for a given array of groupwise losses according to the
MinMaxLoss rules.
"""
evaluate(l::MinMaxLoss, groups::Array{Float64}) = max(groups)

### WEIGHTED LP NORM LOSS ###
mutable struct WeightedLPNormLoss<:WeightedGroupFunctional
    p::Int64
    k::Int64
    weights::Array{Float64}
end

WeightedLPNormLoss(p::Int64, k::Int64; weights::Array{Float64}=fill(1.0/k, k)) = WeightedLPNormLoss(p, k, weights)

"""
# Parameters
- `l::WeightedLPNormLoss`:  The given loss instance.
- `groups::Array{Float64}`: The given groupwise losses.

# Returns
The total error for a given array of groupwise losses according to the
WeightedLPNormLoss rules.
"""
function evaluate(l::WeightedLPNormLoss, groups::Array{Float64})
    validate_weights(l)
    total_err = sum([groups[i]^p * l.weights[i] for i=1:l.k])^(1.0/p)
    total_err
end

# Penalised learning
mutable struct PenalisedLearningLoss<:WeightedGroupFunctional
    k::Int64
    λ::Float64
    weights::Array{Float64}
    d::Function{Float64, Float64}
end
PenalisedLearningLoss(k::Int64, λ::Float64; weights::Array{Float64}=fill(1.0/k, k), d::Function{Float64, Float64}=x,y->abs(x - y)) = PenalisedLearningLoss(k, λ, weights, d)

"""
# Parameters
- `l::PenalisedLearningLoss`: The given loss instance.
- `groups::Array{Float64}`:   The given groupwise losses.

# Returns
The total error for a given array of groupwise losses according to the
PenalisedLearningLoss rules.
"""
function evaluate(l::PenalisedLearningLoss, groups::Array{Float64})
    validate_weights(l)
    @assert l.λ > 0
    err = sum([groups[i] * l.weights[i] for i=1:l.k])
    penalty = l.λ * sum([d(groups[i], groups[j]) for i=1:l.k, j=1:l.k])
    err + penalty
end