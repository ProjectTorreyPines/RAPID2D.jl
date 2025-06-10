using Test
using RAPID2D

if !isempty(ARGS) && !(length(ARGS) == 1 && ARGS[1] == "")
    println(ARGS)
    for testfile in ARGS
        # Skip ADIOS tests on Windows due to compatibility issues
        if Sys.iswindows() && contains(testfile, "adios_io_test.jl")
            @warn "Skipping ADIOS I/O test on Windows: $testfile"
            continue
        end

        @info "Running test file: $testfile"
        include(testfile)
    end
else
    # Default behavior: run all tests
    include("io/wall_io_test.jl")

    # Skip ADIOS tests on Windows due to compatibility issues
    if !Sys.iswindows()
        include("io/adios_io_test.jl")
    else
        @warn "Skipping ADIOS I/O tests on Windows due to known compatibility issues"
    end

    include("utils/utils_test.jl")
    include("utils/green_function_test.jl")

    include("diagnostics/snapshots_test.jl")

    include("fields/calculate_B_fields_test.jl")

    include("numerics/numerics_test.jl")
    include("numerics/operators_test.jl")
    include("numerics/discretized_operator_test.jl")

    include("reactions/RRCs_test.jl")
    include("reactions/electron_Xsec_test.jl")

    include("coils/coils_basic_test.jl")
    include("coils/coils_operations_test.jl")
    include("coils/coils_voltage_functions_test.jl")
    include("coils/coils_initialization_test.jl")

    include("physics/physics_test.jl")
end
