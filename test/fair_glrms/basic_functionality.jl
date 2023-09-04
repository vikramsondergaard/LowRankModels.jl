using LowRankModels, Random

# Test basic functionality of fair_glrms.jl
Random.seed!(1)
m, n, k, s = 100, 100, 5, 100 * 100;
# Matrix to encode
X_real, Y_real = randn(m, k), randn(k, n);
A = X_real * Y_real

# The first column of A is the protected class.
#
# Want to stack the distribution of this column towards some sort of
# "majority". In this case this means that 1 is the majority and 2, 3, and 4
# are minorities to increasing degrees.
for i=1:size(A)[1] A[i, 1] = rand([1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4]) end

# losses = fill(QuadLoss(), n)
# losses[1] = PoissonLoss() # this is the loss for the categorical protected category
losses = vcat([PoissonLoss()], fill(QuadLoss(), n - 1))
rx, ry = ZeroReg(), ZeroReg()

protected_characteristic_idx = 1
num_protected_categories = 4

groups = partition_groups(A, protected_characteristic_idx)

fglrm = FairGLRM(A,losses,rx,ry,k,protected_characteristic_idx,WeightedLogSumExponentialLoss(1, [length(groups[i]) / m for i=1:num_protected_categories]),
                 scale=false, offset=false, X=randn(k,m), Y=randn(k,n));

p = Params(1, max_iter=200, abs_tol=0.0000001, min_stepsize=0.001)
@time X,Y,ch = fit!(fglrm, params=p, verbose=false);
Ah = X'*Y;
p.abs_tol > abs(norm(A-Ah)^2 - ch.objective[end])

function validate_folds(trf,tre,tsf,tse)
	for i=1:length(trf)
		if length(intersect(Set(trf[i]), Set(tsf[i]))) > 0
			println("Error on example $i: train and test sets overlap")
		end
	end
	for i=1:length(tre)
		if length(intersect(Set(tre[i]), Set(tse[i]))) > 0
			println("Error on feature $i: train and test sets overlap")
		end
	end
	true
end

obs = LowRankModels.flatten_observations(glrm.observed_features)
folds = LowRankModels.getfolds(obs, 5, size(glrm.A)..., do_check = false)
for i in 1:length(folds)
	@assert validate_folds(folds[i]...)
end