using LowRankModels
using CSV
using DataFrames

import XLSX
import JSON

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

"""
    test_adult()

Test that partition_groups() correctly partitions the Adult dataset based on
gender.
"""
function test_adult()
    # Path to the Adult dataset
    datapath = "/Users/vikramsondergaard/honours/data/adult/adult.data"
    # Read the CSV file and convert it to a matrix
    adult_csv = CSV.read(datapath, DataFrame, header=false)
    A = convert(Matrix, adult_csv)
    # Path to a JSON file with the expected partition of the groups
    expected_path = "/Users/vikramsondergaard/honours/LowRankModels.jl/data/adult_partition_groups.json"
    # Read the JSON file: this can stay as a Dict
    expected = JSON.parsefile(expected_path)
    # Gender is the 10th column in the Adult data set
    index_of_gender = 10
    test(A, index_of_gender, expected)
    println("Passed test_adult()!")
end

function test_ad_observatory()
    # Path to the (condensed) Ad Observatory dataset
    datapath = "/Users/vikramsondergaard/honours/LowRankModels.jl/data/ad_obs_observer_data.csv"
    # Read in the Ad Observatory data and convert it to a matrix
    ad_obs_data = CSV.read(datapath, DataFrame, header=1)
    A = convert(Matrix, ad_obs_data)
    # Path to a JSON file with the expected partition of the groups
    expected_path = "/Users/vikramsondergaard/honours/LowRankModels.jl/data/ad_obs_partition_groups.json"
    # Read the JSON file: this can stay as a Dict
    expected = JSON.parsefile(expected_path)
    # "Observer Income" is the 7th column in the Ad Observatory data set
    index_of_income = 7
    test(A, index_of_income, expected)
    println("Passed test_ad_observatory()!")
end

test_trivial()
test_adult()
test_ad_observatory()