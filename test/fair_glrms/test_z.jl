using LowRankModels

function test(loss::L where L<:Loss, u::Real, a::Number, magnitude_Ωₖ::Int64, zexp::Float64)
    zout = z(loss, u, a, magnitude_Ωₖ)
    print("zexp: "); display(zexp)
    print("zout: "); display(zout)
    @assert isapprox(zout, zexp, atol=0.001)
end

losses = [OvALoss(3, bin_loss=HingeLoss()), OvALoss(4, bin_loss=HingeLoss()),
    QuadLoss(), QuadLoss(), OrdinalHingeLoss(5)]

A  = [1   2  -0.16297   -1.59880   1
      1   1  -0.15945   -0.34759   1
      2   4   0.28960   -1.10336   3
      2   3   0.58209    2.83976   4
      3   3   1.96867    0.52227   2]

XY = [1   3   0.44340   -0.01352   1
      1   1  -0.15945   -0.34759   1
      2   4   0.28960   -1.01740   1
      3   3   0.24535   -0.69556   5
      1   2   0.71825    0.52227   2]

Z  = [1 2.5 0.36768/2  2.51311/2   0
      1 1.5         0          0   0
      1 1.5         0  0.00739/2 1.5
      2 1.5 0.11340/2 12.49849/2 0.5
      4   5   1.56355          0   0]

magnitudes = [2, 2, 2, 2, 1]

for r=1:5
    for c=1:5
        loss = losses[c]
        u = XY[r, c]
        a = A[r, c]
        if !isa(loss, DiffLoss)
            u = Int(u)
            a = Int(a)
        end
        magnitude_Ωₖ = magnitudes[r]
        zexp = Z[r, c]
        println("Current entry: ($r, $c)")
        test(loss, u, a, magnitude_Ωₖ, zexp)
    end
end
println("Completed test_z!")