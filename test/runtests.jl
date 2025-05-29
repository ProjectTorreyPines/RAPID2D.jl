using Test
using RAPID2D

if !isempty(ARGS) && !(length(ARGS) == 1 && ARGS[1] == "")
    println(ARGS)
    for testfile in ARGS
        @info "Running test file: $testfile"
        include(testfile)
    end
else
    # Default behavior: run all tests
    include("io/io_test.jl")
    include("utils/utils_test.jl")

    include("diagnostics/snapshots_test.jl")

    include("numerics/numerics_test.jl")
    include("numerics/operators_test.jl")
    include("numerics/discretized_operator_test.jl")

    include("reactions/RRCs_test.jl")
    include("reactions/electron_Xsec_test.jl")

    include("physics/physics_test.jl")
end
