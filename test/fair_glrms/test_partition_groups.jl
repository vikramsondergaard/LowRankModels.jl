using LowRankModels
using CSV
using DataFrames

import XLSX
import JSON

ADULT_HEADERS = ["age", "workclass", "fnlwgt", "education", "education-num",
                 "marital-status", "occupation", "relationship", "race", "sex",
                 "capital-gain", "capital-loss", "hours-per-week",
                 "native-country", "yearly-income"]

"""
    test(A, s::Int, expected)

Generic test function for the function partition_groups() from fair_glrms.jl.
It takes a data matrix `A` and the column index `s` of a protected
characteristic, as well as the expected output `expected`.
"""
function test(A, s::Int, expected)
    # Want the element types to be the same (otherwise this would definitely be
    # incorrect!)
    # @assert eltype(A) == eltype(keys(expected))
    # Partition the group using partition_groups()
    groups = partition_groups(A, s)
    # Get the unique protected characteristic categories from the data set
    unique_categories = keys(groups)
    # Get the keys from the expected output - these should be the same as the
    # unique categories
    observed_categories = keys(expected)
    # First part of checking the keys are equivalent
    @assert length(observed_categories) == length(unique_categories)
    display(unique_categories)
    display(observed_categories)
    # Second part of checking the keys are equivalent
    for unique_cat in unique_categories
        @assert unique_cat in observed_categories
    end
    # Second part of checking the values are equivalent
    for k in observed_categories
        @assert size(groups[k])[1] == size(expected[k])[1]
        for v in expected[k]
            @assert v in groups[k]
        end
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
    # Path to the Adult dataset
    datapath = "/Users/vikramsondergaard/honours/data/adult/adult.data"
    # Read the CSV file
    adult_csv = CSV.read(datapath, DataFrame, header=false)
    A = convert(Matrix, adult_csv)
    # Path to a JSON file with the expected partition of the groups
    expected_path = "/Users/vikramsondergaard/honours/LowRankModels.jl/data/adult_partition_groups.json"
    expected = JSON.parsefile(expected_path)
    index_of_gender = 10
    test(A, index_of_gender, expected)
    println("Passed test_adult()!")
end

function test_ad_observatory()
end

function test_german_credit()
end

test_trivial()
test_adult()