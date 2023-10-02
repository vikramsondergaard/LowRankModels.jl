using LowRankModels, CSV, DataFrames, Statistics

import XLSX
import JSON
import Random

Random.seed!(1)

# Path to the Adult dataset
# datapath = "/Users/vikramsondergaard/honours/data/adult/adult_trimmed.data"
datapath = "/Users/vikramsondergaard/honours/LowRankModels.jl/data/adult/adult_trimmed.data"
# Read the CSV file and convert it to a matrix
adult_csv = CSV.read(datapath, DataFrame, header=1)
A = convert(Matrix, adult_csv)
# Gender is the 10th column in the Adult data set
index_of_gender = 3
n_genders = 2 # (from the data - gender is not actually a binary!)

# Columns of dataset
# age - hours-worked - gender - income - workclass - relationship - race - education-num

p = Params(1, max_iter=200, abs_tol=0.0000001, min_stepsize=0.001)

real_losses = [HuberLoss(), HuberLoss()]

bool_losses = [HingeLoss(), HingeLoss()]

n_workclasses = 9
n_relationships = 6
n_races = 5 # (from the data - there are not five races)
n_educations = 16
cat_losses = [OvALoss(n_workclasses, bin_loss=HingeLoss()),
              OvALoss(n_relationships, bin_loss=HingeLoss()),
              OvALoss(n_races, bin_loss=HingeLoss())]

ord_losses = [OrdinalHingeLoss(n_educations)]

losses = [real_losses..., bool_losses..., cat_losses..., ord_losses...]

groups = partition_groups(A, index_of_gender, n_genders)

m,n = size(A)
k = 4

X_init = randn(k, m)
embedding_dims = n + (n_workclasses - 1) + (n_relationships - 1) + (n_races - 1)
Y_init = randn(k, embedding_dims)

glrm = GLRM(A, losses, QuadReg(0.1), QuadReg(0.1), k, X=deepcopy(X_init), Y=deepcopy(Y_init))
glrmX, glrmY, glrmch = fit!(glrm, params=p, verbose=true)
println("successfully fit vanilla GLRM")

fglrm = FairGLRM(A, losses, QuadReg(0.1), QuadReg(0.1), k, index_of_gender,
    WeightedLogSumExponentialLoss(10^(-6), [Float64(length(groups[i])) / m for i=1:n_genders]),
    X=deepcopy(X_init), Y=deepcopy(Y_init), Z=groups)
fglrmX, fglrmY, fglrmch = fit!(fglrm, params=p, verbose=true)
println("successfully fit fair GLRM")

glrm_XY = glrmX' * glrmY
fglrm_XY = fglrmX' * fglrmY

rows,cols = size(glrm_XY)

for r=1:rows
    for c=1:cols
        println("At row $r, column $c")
        println("The value of the GLRM is: $(glrm_XY[r, c])")
        println("The value of the fair GLRM is: $(fglrm_XY[r, c])")
        @assert isapprox(glrm_XY[r, c], fglrm_XY[r, c], atol=0.1)
    end
end