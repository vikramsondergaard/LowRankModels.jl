include("test.jl")
include("test_vanilla_glrm.jl")
include("test_pca.jl")
test_pca("independence")
test_vanilla_glrm("independence")
test("independence")