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
end
