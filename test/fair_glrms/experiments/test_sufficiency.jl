using CSV
include("test.jl")
include("test_vanilla_glrm.jl")

args = parse_commandline()
if !isempty(args["startX"]) && !isempty(args["startY"])
    glrmX = CSV.read(args["startX"], DataFrame, header=1) |> Matrix
    glrmY = CSV.read(args["startY"], DataFrame, header=1) |> Matrix
else
    glrmX, glrmY = test_vanilla_glrm()
    if !isempty(args["startX"]) glrmX = args["startX"] end
    if !isempty(args["startY"]) glrmY = args["startY"] end
end
test("sufficiency", glrmX, glrmY)