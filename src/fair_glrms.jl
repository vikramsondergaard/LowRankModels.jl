# This code is from the paper [1] "Towards Fair Unsupervised Learning" by
# Buet-Golfouse and Utyagulov. It extends GLRMs towards notions of fairness
# using group functionals in place of the loss function in a GLRM.

"""
    partition_groups(A, s::Int)

Partitions the data matrix `A` into sub-matrices according to their value in
the `s`-th column. For example, given the matrix

[1 2 3
 2 2 1
 3 1 2
 4 0 0]

partition_groups(A, 2) would create three sub-matrices(/vectors). These would be
[[4, 0, 0]]   --> for the category "0"
[[3, 1, 2]]   --> for the category "1"
[[1, 2, 3],
 [2, 2, 1]]   --> for the category "2"

The output is a dictionary, where each unique value `k` in the `s`-th column is
a key in the dictionary, and the corresponding value is a matrix whose rows are
all the rows in `A` whose value in the `s`-th column is `k`.

"""
function partition_groups(A, s::Int)
    # Define the groups. The element type for all keys and values is the
    # element type of `A`.
    groups = Dict{eltype(A), Vector{Array{eltype(A)}}}()
    num_rows = size(A)[1]
    for r in 1:num_rows
        row = A[r, :]
        # Get the unique value `k` (see the docstring for more detail)
        k = row[s]
        # The dictionary doesn't yet have this key - add it to the dictionary
        if !haskey(groups, k)
            groups[k] = [row]
        # The dictionary already has this key - push this row to the existing
        # vector
        else
            push!(groups[k], row)
        end
    end
    groups
end