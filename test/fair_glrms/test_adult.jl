using LowRankModels, CSV, DataFrames, Statistics

import XLSX
import JSON

# Path to the Adult dataset
# datapath = "/Users/vikramsondergaard/honours/data/adult/adult_trimmed.data"
datapath = "/Users/vikram/honours/LowRankModels.jl/data/adult/adult_trimmed.data"
# Read the CSV file and convert it to a matrix
adult_csv = CSV.read(datapath, DataFrame, header=1)
A = convert(Matrix, adult_csv)
# Gender is the 10th column in the Adult data set
index_of_gender = 3
n_genders = 2 # (from the data - gender is not actually a binary!)

real_losses = [HuberLoss(), HuberLoss()]

bool_losses = [HingeLoss(), HingeLoss()]

n_workclasses = 9
n_relationships = 6
n_races = 5 # (from the data - there are not five races)
n_educations = 16
cat_losses = [OvALoss(n_workclasses), OvALoss(n_relationships),
              OvALoss(n_races)]

ord_losses = [OrdinalHingeLoss(n_educations)]

losses = [real_losses..., bool_losses..., cat_losses..., ord_losses...]

groups = partition_groups(A, index_of_gender, n_genders)

m,n = size(A)

for i=1:n
    μ = mean(A[:, i])
    σ² = varm(A[:, i], μ)
    mul!(losses[i], 1.0 / σ²)
end

glrm = GLRM(A, losses, QuadReg(0.1), QuadReg(0.1), 4)
glrmX, glrmY, _ = fit!(glrm, verbose=true)
println("successfully fit vanilla GLRM")

fglrm = FairGLRM(A, losses, QuadReg(0.1), QuadReg(0.1), 4, index_of_gender,
    WeightedLogSumExponentialLoss(1.0 / (10.0^6), [length(groups[i]) / m for i=1:n_genders]))
fglrmX, fglrmY, _ = fit!(fglrm, verbose=true)
println("successfully fit fair GLRM")

glrm_XY = glrmX' * glrmY
fglrm_XY = fglrmX' * fglrmY

rows,cols = size(glrm_XY)

for r=1:rows
    for c=1:cols
        println("At row $r, column $c")
        println("The value of the GLRM is: $(glrm_XY[r, c])")
        println("The value of the fair GLR is: $(fglrm_XY[r, c])")
        @assert isapprox(glrm_XY[r, c], fglrm_XY[r, c], atol=0.1)
    end
end