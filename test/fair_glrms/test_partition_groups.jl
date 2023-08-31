using LowRankModels

"""
    test(A, s::Int, expected)

Generic test function for the function partition_groups() from fair_glrms.jl.
It takes a data matrix `A` and the column index `s` of a protected
characteristic, as well as the expected output `expected`.
"""
function test(A, s::Int, expected)
    # Want the element types to be the same (otherwise this would definitely be
    # incorrect!)
    @assert eltype(A) == eltype(keys(expected))
    # Partition the group using partition_groups()
    groups = partition_groups(A, s)
    # Get the unique protected characteristic categories from the data set
    unique_categories = unique([A[k, s] for k in 1:size(A)[1]])
    # Get the keys from the expected output - these should be the same as the
    # unique categories
    observed_categories = keys(expected)
    # First part of checking the keys are equivalent
    @assert length(observed_categories) == length(unique_categories)
    # Second part of checking the keys are equivalent
    for unique_cat in unique_categories
        @assert unique_cat in observed_categories
    end
    # First part of checking the values are equivalent
    @assert size(A)[1] == sum(size(expected[k])[1] for k in observed_categories)
    # Second part of checking the values are equivalent
    for x in 1:size(A)[1]
        category = A[x, s]
        @assert A[x, :] in expected[category]
    end
end

"""
    test_trivial()

Test the trivial example from the docstring for partition_groups(). See
`fair_glrms.jl` for the exact specification.
"""
function test_trivial()
    A = [1 2 3
         2 2 1
         3 1 2
         4 0 0]
    s = 2
    expected = Dict(0 => [[4, 0, 0]],
                    1 => [[3, 1, 2]],
                    2 => [[1, 2, 3], [2, 2, 1]])
    test(A, s, expected)
    println("Passed test_trivial()!")
end

function test_adult()
end

function test_ad_observatory()
end

function test_german_credit()
end

test_trivial()