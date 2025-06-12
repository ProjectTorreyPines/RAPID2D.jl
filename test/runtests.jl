using Test
using RAPID2D

function find_test_files(dir::String)
    @assert isdir(dir) "Directory does not exist: $dir"
    test_files = String[]
    for (root, _, files) in walkdir(dir)
        for file in files
            if endswith(file, "_test.jl")
                push!(test_files, joinpath(root, file))
            end
        end
    end
    return sort(test_files)  # Sort for deterministic order
end

function include_tests_in_dir(dir::String)
    @assert isdir(dir) "Directory does not exist: $dir"
    test_files = find_test_files(dir)
    for filepath in test_files
        @info "Running test file: $filepath"
        include(filepath)
    end
end

if !isempty(ARGS) && !(length(ARGS) == 1 && ARGS[1] == "")
    println(ARGS)
    for testfile in ARGS
        @info "Running test file: $testfile"
        include(testfile)
    end
else
    # Always run unit tests first
    @info "Running unit tests..."
    if isdir("unit")
        include_tests_in_dir("unit")
    else
        @warn "Unit test directory not found"
    end

    # Conditionally run regression tests
    run_regression = get(ENV, "RAPID_RUN_REGRESSION", "false") == "true"
    if run_regression
        @info "Running regression tests..."
        if isdir("regression")
            include_tests_in_dir("regression")
        else
            @warn "Regression test directory not found"
        end
    else
        @info "Skipping regression tests. Set RAPID_RUN_REGRESSION=true to run them."
    end
end
