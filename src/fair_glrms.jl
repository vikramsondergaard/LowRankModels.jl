# This code is from the paper [1] "Towards Fair Unsupervised Learning" by
# Buet-Golfouse and Utyagulov. It extends GLRMs towards notions of fairness
# using group functionals in place of the loss function in a GLRM.

export FairGLRM, find_group

### FAIR GLRM TYPE
mutable struct FairGLRM<:AbstractGLRM
    A                                   # The data table
    losses::Array{Loss,1}               # array of loss functions
    rx::Array{Regularizer,1}            # Array of regularizers to be applied to each column of X
    ry::Array{Regularizer,1}            # Array of regularizers to be applied to each column of Y
    k::Int                              # Desired rank
    protected_category                  # The protected characteristic with which to separate the data for Z
    group_functional::GroupFunctional   # The group functional that brings together all separate losses
    observed_features::ObsArray         # for each example, an array telling which features were observed
    observed_examples::ObsArray         # for each feature, an array telling in which examples the feature was observed
    X::AbstractArray{Float64,2}         # Representation of data in low-rank space. A ≈ X'Y
    Y::AbstractArray{Float64,2}         # Representation of features in low-rank space. A ≈ X'Y
    Z                                   # A Dict that separates the data table by a given protected characteristic
end

"""
    partition_groups(A, s::Int)

Partitions the data matrix `A` into sub-matrices according to their value in
the `s`-th column. For example, given the matrix

```
[1 2 3
 2 2 1
 3 1 2
 4 0 0]
```

partition_groups(A, 2) would create three sub-matrices(/vectors). These would be
```
[[4, 0, 0]]   --> for the category "0"
[[3, 1, 2]]   --> for the category "1"
[[1, 2, 3],
 [2, 2, 1]]   --> for the category "2"
```

The output is a dictionary, where each unique value `k` in the `s`-th column is
a key in the dictionary, and the corresponding value is a matrix whose rows are
all the rows in `A` whose value in the `s`-th column is `k`.
"""
function partition_groups(A, s::Int)
    # Define the groups. The element type for all keys and values is the
    # element type of `A`.
    groups = Dict{eltype(A[:, s]), Set{Int64}}()
    num_rows = size(A)[1]
    for r in 1:num_rows
        row = A[r, :]
        # Get the unique value `k` (see the docstring for more detail)
        k = row[s]
        # The dictionary doesn't yet have this key - add it to the dictionary
        if !haskey(groups, k)
            groups[k] = Set(r)
        # The dictionary already has this key - push this row to the existing
        # vector
        else
            push!(groups[k], r)
        end
    end
    group_vals = values(groups)
    ordered_group_vals = convert(Vector, group_vals)
    ordered_group_vals
end

function FairGLRM(A, losses::Array, rx::Array, ry::Array, k::Int, s::Int, group_functional::GroupFunctional;
                  X = randn(k,size(A,1)), Y = randn(k,embedding_dim(losses)),
                  Z = partition_groups(A, s),
                  obs = nothing,                                    # [(i₁,j₁), (i₂,j₂), ... (iₒ,jₒ)]
                  observed_features = fill(1:size(A,2), size(A,1)), # [1:n, 1:n, ... 1:n] m times
                  observed_examples = fill(1:size(A,1), size(A,2)), # [1:m, 1:m, ... 1:m] n times
                  offset = false, scale = false,
                  checknan = true, sparse_na = true)
    # Check dimensions of the arguments
    m,n = size(A)
    if length(losses)!=n error("There must be as many losses as there are columns in the data matrix") end
    if length(rx)!=m error("There must be either one X regularizer or as many X regularizers as there are rows in the data matrix") end
    if length(ry)!=n error("There must be either one Y regularizer or as many Y regularizers as there are columns in the data matrix") end
    if size(X)!=(k,m) error("X must be of size (k,m) where m is the number of rows in the data matrix. This is the transpose of the standard notation used in the paper, but it makes for better memory management. \nsize(X) = $(size(X)), size(A) = $(size(A)), k = $k") end
    if size(Y)!=(k,embedding_dim(losses)) error("Y must be of size (k,d) where d is the sum of the embedding dimensions of all the losses. \n(1 for real-valued losses, and the number of categories for categorical losses).") end

    # Determine observed entries of data
    if obs===nothing && sparse_na && isa(A,SparseMatrixCSC)
        obs = findall(!iszero, A) # observed indices (list of CartesianIndices)
    end
    if obs===nothing # if no specified array of tuples, use what was explicitly passed in or the defaults (all)
        # println("no obs given, using observed_features and observed_examples")
        glrm = FairGLRM(A,losses,rx,ry,k, s, group_functional, observed_features, observed_examples, X,Y, Z)
    else # otherwise unpack the tuple list into arrays
        # println("unpacking obs into array")
        glrm = FairGLRM(A,losses,rx,ry,k, s, group_functional, sort_observations(obs,size(A)...)..., X,Y, Z)
    end

    # check to make sure X is properly oriented
    if size(glrm.X) != (k, size(A,1))
        # println("transposing X")
        glrm.X = glrm.X'
    end
    # check none of the observations are NaN
    if checknan
        for i=1:size(A,1)
            for j=glrm.observed_features[i]
                if isnan(A[i,j])
                    error("Observed value in entry ($i, $j) is NaN.")
                end
            end
        end
    end

    if scale # scale losses (and regularizers) so they all have equal variance
        equilibrate_variance!(glrm)
    end
    if offset # don't penalize the offset of the columns
        add_offset!(glrm)
    end
    return glrm
end

function find_group(fglrm::FairGLRM, i::Int64)
    for k=1:size(fglrm.Z)[1]
        if i in fglrm.Z[k] return k end
    end
end