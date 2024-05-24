##############################################################
### copying
##############################################################

import Base.copy
export copy, copy_estimate, GLRM, FairGLRM

for T in :[Loss, Regularizer, AbstractGLRM].args
  @eval function copy(r::$T)
    fieldvals = [getfield(r, f) for f in fieldnames(typeof(r))]
    return typeof(r)(fieldvals...)
  end
end
# points to all the same problem data as the original input GLRM,
# but copies the estimate of the model parameters
function copy_estimate(g::GLRM)
  return GLRM(g.A,g.losses,g.rx,g.ry,g.k,
              g.observed_features,g.observed_examples,
              copy(g.X),copy(g.Y))
end
# domains are struct, so this is ok
copy(d::Domain) = d

##############################################################
### fill singleton losses and regularizers to the right shapes
##############################################################

# fill an array of length n with copies of the object foo
fillcopies(foo, n::Int; arraytype=typeof(foo)) = arraytype[copy(foo) for i=1:n]

# singleton loss:
GLRM(A, loss::Loss, rx::Array, ry::Regularizer, k::Int; kwargs...) =
    GLRM(A, fillcopies(loss, size(A, 2), arraytype=Loss), rx, fillcopies(ry, size(A, 2), arraytype=Regularizer), k; kwargs...)
GLRM(A, loss::Loss, rx::Regularizer, ry::Array, k::Int; kwargs...) =
    GLRM(A, fillcopies(loss, size(A, 2), arraytype=Loss), fillcopies(rx, size(A, 1), arraytype=Regularizer), ry, k; kwargs...)
GLRM(A, loss::Loss, rx::Array, ry::Array, k::Int; kwargs...) =
    GLRM(A, fillcopies(loss, size(A, 2), arraytype=Loss), rx, ry, k; kwargs...)

# FairGLRM(A, loss::Loss, rx::Array, ry::Regularizer, rkx::Array, rky::Array, k::Int, s, group_functional::GroupFunctional; kwargs...) =
#     FairGLRM(A, fillcopies(loss, size(A, 2), arraytype=Loss), rx, fillcopies(ry, size(A, 2), arraytype=Regularizer), rkx, rky, k, s, group_functional; kwargs...)
# FairGLRM(A, loss::Loss, rx::Regularizer, ry::Array, rkx::Array, rky::Array, k::Int, s, group_functional::GroupFunctional; kwargs...) =
#     FairGLRM(A, fillcopies(loss, size(A, 2), arraytype=Loss), fillcopies(rx, size(A, 1), arraytype=Regularizer), ry, rkx, rky, k, s, group_functional; kwargs...)
# FairGLRM(A, loss::Loss, rx::Array, ry::Array, rkx::Array, rky::Array, k::Int, s, group_functional::GroupFunctional; kwargs...) =
#     FairGLRM(A, fillcopies(loss, size(A, 2), arraytype=Loss), rx, ry, rkx, rky, k, s, group_functional; kwargs...)
# FairGLRM(A, loss::Loss, rx::Array, ry::Array, rkx::ColumnRegularizer, rky::Array, k::Int, s, group_functional::GroupFunctional; kwargs...) =
#     FairGLRM(A, fillcopies(loss, size(A, 2), arraytype=Loss), rx, ry, fillcopies(rkx, k), rky, k, s, group_functional; kwargs...)
# FairGLRM(A, loss::Loss, rx::Array, ry::Array, rkx::Array, rky::ColumnRegularizer, k::Int, s, group_functional::GroupFunctional; kwargs...) =
#     FairGLRM(A, fillcopies(loss, size(A, 2), arraytype=Loss), rx, ry, rkx, fillcopies(rky, k), k, s, group_functional; kwargs...)
# FairGLRM(A, loss::Loss, rx::Array, ry::Array, rkx::ColumnRegularizer, rky::ColumnRegularizer, k::Int, s, group_functional::GroupFunctional; kwargs...) =
#     FairGLRM(A, fillcopies(loss, size(A, 2), arraytype=Loss), rx, ry, fillcopies(rkx, k), fillcopies(rky, k), k, s, group_functional; kwargs...)
# FairGLRM(A, loss::Loss, rx::Array, ry::Array, rkx::Array, rky::ColumnRegularizer, k::Int, s, group_functional::GroupFunctional; kwargs...) =
#     FairGLRM(A, fillcopies(loss, size(A, 2), arraytype=Loss), rx, ry, rkx, fillcopies(rky, k), k, s, group_functional; kwargs...)

# singleton regularizer on x and/or y:
GLRM(A, losses::Array, rx::Regularizer, ry::Array, k::Int; kwargs...) =
    GLRM(A, losses, fillcopies(rx, size(A, 1), arraytype=Regularizer), ry, k::Int; kwargs...)
GLRM(A, losses::Array, rx::Array, ry::Regularizer, k::Int; kwargs...) =
    GLRM(A, losses, rx, fillcopies(ry, size(A, 2), arraytype=Regularizer), k::Int; kwargs...)
GLRM(A, losses::Array, rx::Regularizer, ry::Regularizer, k::Int; kwargs...) =
    GLRM(A, losses, fillcopies(rx, size(A, 1), arraytype=Regularizer), fillcopies(ry, size(A, 2), arraytype=Regularizer), k::Int; kwargs...)

# FairGLRM(A, losses::Array, rx::Regularizer, ry::Array, k::Int, s, group_functional::GroupFunctional; kwargs...) =
#     FairGLRM(A, losses, fillcopies(rx, size(A, 1), arraytype=Regularizer), ry, k::Int, s, group_functional::GroupFunctional; kwargs...)
# FairGLRM(A, losses::Array, rx::Array, ry::Regularizer, k::Int, s, group_functional::GroupFunctional; kwargs...) =
#     FairGLRM(A, losses, rx, fillcopies(ry, size(A, 2), arraytype=Regularizer), k::Int, s, group_functional::GroupFunctional; kwargs...)
# FairGLRM(A, losses::Array, rx::Regularizer, ry::Regularizer, k::Int, s, group_functional::GroupFunctional; kwargs...) =
#     FairGLRM(A, losses, fillcopies(rx, size(A, 1), arraytype=Regularizer), fillcopies(ry, size(A, 2), arraytype=Regularizer), k::Int, s, group_functional::GroupFunctional; kwargs...)

# singleton everything
GLRM(A, loss::Loss, rx::Regularizer, ry::Regularizer, k::Int; kwargs...) =
    GLRM(A, fillcopies(loss, size(A, 2), arraytype=Loss), fillcopies(rx, size(A, 1), arraytype=Regularizer), fillcopies(ry, size(A, 2), arraytype=Regularizer), k::Int; kwargs...)

#FairGLRM(A, loss::Loss, rx::Regularizer, ry::Regularizer, k::Int, s, group_functional::GroupFunctional; kwargs...) =
   # FairGLRM(A, fillcopies(loss, size(A, 2), arraytype=Loss), fillcopies(rx, size(A, 1), arraytype=Regularizer), fillcopies(ry, size(A, 2), arraytype=Regularizer), k::Int, s, group_functional::GroupFunctional; kwargs...)

FairGLRM(A, loss::Loss, rx::Array, ry::Array, rkx::Array, rky::Array, k::Int, s, group_functional::GroupFunctional; kwargs...) =
   FairGLRM(A, fillcopies(loss, size(A, 2), arraytype=Loss), rx, ry, rkx, rky, k, s, group_functional; kwargs...)
FairGLRM(A, loss::Loss, rx::Array, ry::Array, rkx::Array, rky::ColumnRegularizer, k::Int, s, group_functional::GroupFunctional; kwargs...) =
   FairGLRM(A, fillcopies(loss, size(A, 2), arraytype=Loss), rx, ry, rkx, fillcopies(rky, k, arraytype=ColumnRegularizer), k, s, group_functional; kwargs...)
FairGLRM(A, loss::Loss, rx::Array, ry::Array, rkx::ColumnRegularizer, rky::Array, k::Int, s, group_functional::GroupFunctional; kwargs...) =
   FairGLRM(A, fillcopies(loss, size(A, 2), arraytype=Loss), rx, ry, fillcopies(rkx, k, arraytype=ColumnRegularizer), rky, k, s, group_functional; kwargs...)
FairGLRM(A, loss::Loss, rx::Array, ry::Array, rkx::ColumnRegularizer, rky::ColumnRegularizer, k::Int, s, group_functional::GroupFunctional; kwargs...) =
   FairGLRM(A, fillcopies(loss, size(A, 2), arraytype=Loss), rx, ry, fillcopies(rkx, k, arraytype=ColumnRegularizer), fillcopies(rky, k, arraytype=ColumnRegularizer), k, s, group_functional; kwargs...)
FairGLRM(A, loss::Loss, rx::Array, ry::Regularizer, rkx::Array, rky::Array, k::Int, s, group_functional::GroupFunctional; kwargs...) =
   FairGLRM(A, fillcopies(loss, size(A, 2), arraytype=Loss), rx, fillcopies(ry, size(A, 2), arraytype=Regularizer), rkx, rky, k, s, group_functional; kwargs...)
FairGLRM(A, loss::Loss, rx::Array, ry::Regularizer, rkx::Array, rky::ColumnRegularizer, k::Int, s, group_functional::GroupFunctional; kwargs...) =
   FairGLRM(A, fillcopies(loss, size(A, 2), arraytype=Loss), rx, fillcopies(ry, size(A, 2), arraytype=Regularizer), rkx, fillcopies(rky, k, arraytype=ColumnRegularizer), k, s, group_functional; kwargs...)
FairGLRM(A, loss::Loss, rx::Array, ry::Regularizer, rkx::ColumnRegularizer, rky::Array, k::Int, s, group_functional::GroupFunctional; kwargs...) =
   FairGLRM(A, fillcopies(loss, size(A, 2), arraytype=Loss), rx, fillcopies(ry, size(A, 2), arraytype=Regularizer), fillcopies(rkx, k, arraytype=ColumnRegularizer), rky, k, s, group_functional; kwargs...)
FairGLRM(A, loss::Loss, rx::Array, ry::Regularizer, rkx::ColumnRegularizer, rky::ColumnRegularizer, k::Int, s, group_functional::GroupFunctional; kwargs...) =
   FairGLRM(A, fillcopies(loss, size(A, 2), arraytype=Loss), rx, fillcopies(ry, size(A, 2), arraytype=Regularizer), fillcopies(rkx, k, arraytype=ColumnRegularizer), fillcopies(rky, k, arraytype=ColumnRegularizer), k, s, group_functional; kwargs...)
FairGLRM(A, loss::Loss, rx::Regularizer, ry::Array, rkx::Array, rky::Array, k::Int, s, group_functional::GroupFunctional; kwargs...) =
   FairGLRM(A, fillcopies(loss, size(A, 2), arraytype=Loss), fillcopies(rx, size(A, 1), arraytype=Regularizer), ry, rkx, rky, k, s, group_functional; kwargs...)
FairGLRM(A, loss::Loss, rx::Regularizer, ry::Array, rkx::Array, rky::ColumnRegularizer, k::Int, s, group_functional::GroupFunctional; kwargs...) =
   FairGLRM(A, fillcopies(loss, size(A, 2), arraytype=Loss), fillcopies(rx, size(A, 1), arraytype=Regularizer), ry, rkx, fillcopies(rky, k, arraytype=ColumnRegularizer), k, s, group_functional; kwargs...)
FairGLRM(A, loss::Loss, rx::Regularizer, ry::Array, rkx::ColumnRegularizer, rky::Array, k::Int, s, group_functional::GroupFunctional; kwargs...) =
   FairGLRM(A, fillcopies(loss, size(A, 2), arraytype=Loss), fillcopies(rx, size(A, 1), arraytype=Regularizer), ry, fillcopies(rkx, k, arraytype=ColumnRegularizer), rky, k, s, group_functional; kwargs...)
FairGLRM(A, loss::Loss, rx::Regularizer, ry::Array, rkx::ColumnRegularizer, rky::ColumnRegularizer, k::Int, s, group_functional::GroupFunctional; kwargs...) =
   FairGLRM(A, fillcopies(loss, size(A, 2), arraytype=Loss), fillcopies(rx, size(A, 1), arraytype=Regularizer), ry, fillcopies(rkx, k, arraytype=ColumnRegularizer), fillcopies(rky, k, arraytype=ColumnRegularizer), k, s, group_functional; kwargs...)
FairGLRM(A, loss::Loss, rx::Regularizer, ry::Regularizer, rkx::Array, rky::Array, k::Int, s, group_functional::GroupFunctional; kwargs...) =
   FairGLRM(A, fillcopies(loss, size(A, 2), arraytype=Loss), fillcopies(rx, size(A, 1), arraytype=Regularizer), fillcopies(ry, size(A, 2), arraytype=Regularizer), rkx, rky, k, s, group_functional; kwargs...)
FairGLRM(A, loss::Loss, rx::Regularizer, ry::Regularizer, rkx::Array, rky::ColumnRegularizer, k::Int, s, group_functional::GroupFunctional; kwargs...) =
   FairGLRM(A, fillcopies(loss, size(A, 2), arraytype=Loss), fillcopies(rx, size(A, 1), arraytype=Regularizer), fillcopies(ry, size(A, 2), arraytype=Regularizer), rkx, fillcopies(rky, k, arraytype=ColumnRegularizer), k, s, group_functional; kwargs...)
FairGLRM(A, loss::Loss, rx::Regularizer, ry::Regularizer, rkx::ColumnRegularizer, rky::Array, k::Int, s, group_functional::GroupFunctional; kwargs...) =
   FairGLRM(A, fillcopies(loss, size(A, 2), arraytype=Loss), fillcopies(rx, size(A, 1), arraytype=Regularizer), fillcopies(ry, size(A, 2), arraytype=Regularizer), fillcopies(rkx, k, arraytype=ColumnRegularizer), rky, k, s, group_functional; kwargs...)
FairGLRM(A, loss::Loss, rx::Regularizer, ry::Regularizer, rkx::ColumnRegularizer, rky::ColumnRegularizer, k::Int, s, group_functional::GroupFunctional; kwargs...) =
   FairGLRM(A, fillcopies(loss, size(A, 2), arraytype=Loss), fillcopies(rx, size(A, 1), arraytype=Regularizer), fillcopies(ry, size(A, 2), arraytype=Regularizer), fillcopies(rkx, k, arraytype=ColumnRegularizer), fillcopies(rky, k, arraytype=ColumnRegularizer), k, s, group_functional; kwargs...)
#FairGLRM(A, losses::Array, rx::Array, ry::Array, rkx::Array, rky::Array, k::Int, s, group_functional::GroupFunctional; kwargs...) =
   #FairGLRM(A, losses, rx, ry, rkx, rky, k, s, group_functional; kwargs...)
FairGLRM(A, losses::Array, rx::Array, ry::Array, rkx::Array, rky::ColumnRegularizer, k::Int, s, group_functional::GroupFunctional; kwargs...) =
   FairGLRM(A, losses, rx, ry, rkx, fillcopies(rky, k, arraytype=ColumnRegularizer), k, s, group_functional; kwargs...)
FairGLRM(A, losses::Array, rx::Array, ry::Array, rkx::ColumnRegularizer, rky::Array, k::Int, s, group_functional::GroupFunctional; kwargs...) =
   FairGLRM(A, losses, rx, ry, fillcopies(rkx, k, arraytype=ColumnRegularizer), rky, k, s, group_functional; kwargs...)
FairGLRM(A, losses::Array, rx::Array, ry::Array, rkx::ColumnRegularizer, rky::ColumnRegularizer, k::Int, s, group_functional::GroupFunctional; kwargs...) =
   FairGLRM(A, losses, rx, ry, fillcopies(rkx, k, arraytype=ColumnRegularizer), fillcopies(rky, k, arraytype=ColumnRegularizer), k, s, group_functional; kwargs...)
FairGLRM(A, losses::Array, rx::Array, ry::Regularizer, rkx::Array, rky::Array, k::Int, s, group_functional::GroupFunctional; kwargs...) =
   FairGLRM(A, losses, rx, fillcopies(ry, size(A, 2), arraytype=Regularizer), rkx, rky, k, s, group_functional; kwargs...)
FairGLRM(A, losses::Array, rx::Array, ry::Regularizer, rkx::Array, rky::ColumnRegularizer, k::Int, s, group_functional::GroupFunctional; kwargs...) =
   FairGLRM(A, losses, rx, fillcopies(ry, size(A, 2), arraytype=Regularizer), rkx, fillcopies(rky, k, arraytype=ColumnRegularizer), k, s, group_functional; kwargs...)
FairGLRM(A, losses::Array, rx::Array, ry::Regularizer, rkx::ColumnRegularizer, rky::Array, k::Int, s, group_functional::GroupFunctional; kwargs...) =
   FairGLRM(A, losses, rx, fillcopies(ry, size(A, 2), arraytype=Regularizer), fillcopies(rkx, k, arraytype=ColumnRegularizer), rky, k, s, group_functional; kwargs...)
FairGLRM(A, losses::Array, rx::Array, ry::Regularizer, rkx::ColumnRegularizer, rky::ColumnRegularizer, k::Int, s, group_functional::GroupFunctional; kwargs...) =
   FairGLRM(A, losses, rx, fillcopies(ry, size(A, 2), arraytype=Regularizer), fillcopies(rkx, k, arraytype=ColumnRegularizer), fillcopies(rky, k, arraytype=ColumnRegularizer), k, s, group_functional; kwargs...)
FairGLRM(A, losses::Array, rx::Regularizer, ry::Array, rkx::Array, rky::Array, k::Int, s, group_functional::GroupFunctional; kwargs...) =
   FairGLRM(A, losses, fillcopies(rx, size(A, 1), arraytype=Regularizer), ry, rkx, rky, k, s, group_functional; kwargs...)
FairGLRM(A, losses::Array, rx::Regularizer, ry::Array, rkx::Array, rky::ColumnRegularizer, k::Int, s, group_functional::GroupFunctional; kwargs...) =
   FairGLRM(A, losses, fillcopies(rx, size(A, 1), arraytype=Regularizer), ry, rkx, fillcopies(rky, k, arraytype=ColumnRegularizer), k, s, group_functional; kwargs...)
FairGLRM(A, losses::Array, rx::Regularizer, ry::Array, rkx::ColumnRegularizer, rky::Array, k::Int, s, group_functional::GroupFunctional; kwargs...) =
   FairGLRM(A, losses, fillcopies(rx, size(A, 1), arraytype=Regularizer), ry, fillcopies(rkx, k, arraytype=ColumnRegularizer), rky, k, s, group_functional; kwargs...)
FairGLRM(A, losses::Array, rx::Regularizer, ry::Array, rkx::ColumnRegularizer, rky::ColumnRegularizer, k::Int, s, group_functional::GroupFunctional; kwargs...) =
   FairGLRM(A, losses, fillcopies(rx, size(A, 1), arraytype=Regularizer), ry, fillcopies(rkx, k, arraytype=ColumnRegularizer), fillcopies(rky, k, arraytype=ColumnRegularizer), k, s, group_functional; kwargs...)
FairGLRM(A, losses::Array, rx::Regularizer, ry::Regularizer, rkx::Array, rky::Array, k::Int, s, group_functional::GroupFunctional; kwargs...) =
   FairGLRM(A, losses, fillcopies(rx, size(A, 1), arraytype=Regularizer), fillcopies(ry, size(A, 2), arraytype=Regularizer), rkx, rky, k, s, group_functional; kwargs...)
FairGLRM(A, losses::Array, rx::Regularizer, ry::Regularizer, rkx::Array, rky::ColumnRegularizer, k::Int, s, group_functional::GroupFunctional; kwargs...) =
   FairGLRM(A, losses, fillcopies(rx, size(A, 1), arraytype=Regularizer), fillcopies(ry, size(A, 2), arraytype=Regularizer), rkx, fillcopies(rky, k, arraytype=ColumnRegularizer), k, s, group_functional; kwargs...)
FairGLRM(A, losses::Array, rx::Regularizer, ry::Regularizer, rkx::ColumnRegularizer, rky::Array, k::Int, s, group_functional::GroupFunctional; kwargs...) =
   FairGLRM(A, losses, fillcopies(rx, size(A, 1), arraytype=Regularizer), fillcopies(ry, size(A, 2), arraytype=Regularizer), fillcopies(rkx, k, arraytype=ColumnRegularizer), rky, k, s, group_functional; kwargs...)
FairGLRM(A, losses::Array, rx::Regularizer, ry::Regularizer, rkx::ColumnRegularizer, rky::ColumnRegularizer, k::Int, s, group_functional::GroupFunctional; kwargs...) =
   FairGLRM(A, losses, fillcopies(rx, size(A, 1), arraytype=Regularizer), fillcopies(ry, size(A, 2), arraytype=Regularizer), fillcopies(rkx, k, arraytype=ColumnRegularizer), fillcopies(rky, k, arraytype=ColumnRegularizer), k, s, group_functional; kwargs...)