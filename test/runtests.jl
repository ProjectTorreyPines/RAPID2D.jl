using Test
using RAPID2D

function include_tests_in_dir(dir::String)
    @assert isdir(dir) "Directory does not exist: $dir"

    test_files = filter(f -> endswith(f, "_test.jl"), readdir(dir))
    for file in sort(test_files)  # sort for deterministic order
        filepath = joinpath(dir, file)
        @info "Running test file: $filepath"
        include(filepath)
    end
end

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

    include_tests_in_dir("utils")
    include_tests_in_dir("diagnostics")
    include_tests_in_dir("numerics")
    include_tests_in_dir("reactions")
    include_tests_in_dir("coils")

    include("diagnostics/snapshots_test.jl")

    include("fields/calculate_B_fields_test.jl")

    include("physics/physics_test.jl")
end
