using Test
using RAPID2D

if !isempty(ARGS)
    for testfile in ARGS
        @info "Running test file: $testfile"
        include(testfile)
    end
else
    # Default behavior: run all tests
    include("io/io_test.jl")
    include("utils/utils_test.jl")
    include("numeric/numeric_test.jl")
    include("reactions/RRCs_test.jl")

    include("physics/convection_test.jl")
    include("physics/diffusion_test.jl")
    include("physics/physics_test.jl")
end
