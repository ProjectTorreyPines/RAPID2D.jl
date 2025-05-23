# Performance benchmark comparing SparseMatrixCSC vs DiscretizedOperator operations
using RAPID2D
using BenchmarkTools
using SparseArrays
using Random
using Printf
using LinearAlgebra

"""
    generate_test_sparse_matrix(n, sparsity_level, rng=Random.default_rng())

Generate a random sparse matrix of size n×n with the specified sparsity level.
Returns a SparseMatrixCSC{Float64, Int}.
"""
function generate_test_sparse_matrix(n, sparsity_level, rng=Random.default_rng())
    # Calculate number of non-zero elements
    nnz = round(Int, n * n * sparsity_level)

    # Generate random indices and values
    I = rand(rng, 1:n, nnz)
    J = rand(rng, 1:n, nnz)
    V = randn(rng, nnz)

    # Create sparse matrix
    return sparse(one(Float64)*I, J, V, n, n)
end

"""
    create_discretized_operator(matrix::SparseMatrixCSC{Float64, Int})

Create a DiscretizedOperator from a sparse matrix.
"""
function create_discretized_operator(matrix::SparseMatrixCSC{Float64, Int})
    # Calculate 2D dimensions from matrix size
    n = size(matrix, 1)
    # Assume square dimensions for simplicity
    each_size = Int(sqrt(n))
    # Create DiscretizedOperator with the sparse matrix
    return DiscretizedOperator(dims_rz=(each_size, each_size), matrix=matrix)
end

"""
    run_benchmarks(sizes, sparsity_level)

Run performance benchmarks comparing sparse matrix operations with DiscretizedOperator operations.
The benchmark tests various operations including basic arithmetic, element-wise operations and broadcasting.

Parameters:
- sizes: Array of matrix sizes to test (n where matrix is n×n)
- sparsity_level: Fraction of non-zero elements in the matrix
"""
function run_benchmarks(sizes, sparsity_level)
    println("Running performance benchmarks...")
    println("==================================")

    # Set random seed for reproducibility
    rng = Random.MersenneTwister(42)

    # Define operations to benchmark
    operations = [
        ("Linear system solution (A\\b)",
            (A, b) -> A \ b,
            (A, b) -> A \ b),

        ("Addition (A + B)",
            (A, B) -> A + B,
            (A, B) -> A + B),

        ("Subtraction (A - B)",
            (A, B) -> A - B,
            (A, B) -> A - B),

        ("Scalar multiplication (2.0 * A)",
            A -> 2.0 * A,
            A -> 2.0 * A),

        ("Matrix multiplication (A * B)",
            (A, B) -> A * B,
            (A, B) -> A * B),

        ("Element-wise multiplication (A .* B)",
            (A, B) -> A .* B,
            (A, B) -> A .* B),

        ("Element-wise division (A ./ (B + 0.1))",
            (A, B) -> A ./ (B .+ 0.1),
            (A, B) -> A ./ (B .+ 0.1)),

        ("Element-wise power (A .^ 2)",
            A -> A .^ 2,
            A -> A .^ 2),

        ("Simple broadcast (@. C = A + B)",
            (A, B) -> (@. A + B),
            (A, B) -> (@. A + B)),

        ("Complex broadcast (@. C = 2.0 * A + 0.5 * B)",
            (A, B) -> (@. 2.0 * A + 0.5 * B),
            (A, B) -> (@. 2.0 * A + 0.5 * B)),

        ("Nested broadcast (@. C = (A + B)^2)",
            (A, B) -> (@. (A + B)^2),
            (A, B) -> (@. (A + B)^2)),

        # These are combined operations
        ("Chained operations ((A * B) + C)",
            (A, B, C) -> (A * B) + C,
            (A, B, C) -> (A * B) + C),

        ("Mixed operations (A * (B .* C))",
            (A, B, C) -> A * (B .* C),
            (A, B, C) -> A * (B .* C))
    ]

    # Run benchmarks for different matrix sizes
    for size in sizes
        println("\nMatrix size: $(size)×$(size) (total elements: $(size^2)), sparsity: $(sparsity_level*100)%")
        println("-------------------------------------------------------------------------")

        # Generate sparse matrices
        A_sparse = generate_test_sparse_matrix(size^2, sparsity_level, rng)
        B_sparse = generate_test_sparse_matrix(size^2, sparsity_level, rng)
        C_sparse = generate_test_sparse_matrix(size^2, sparsity_level, rng)

        A_sparse .+= sparse(one(Float64)*I, size^2, size^2)
        B_sparse .+= sparse(one(Float64)*I, size^2, size^2)
        C_sparse .+= sparse(one(Float64)*I, size^2, size^2)

        # Generate random vector for solving linear systems (A\b)
        b_vector = randn(rng, size^2)

        # Create DiscretizedOperator objects
        A_dop = create_discretized_operator(copy(A_sparse))
        B_dop = create_discretized_operator(copy(B_sparse))
        C_dop = create_discretized_operator(copy(C_sparse))

        # Print header for the comparison table
        @printf("%-40s | %-20s | %-20s | %-10s\n", "Operation", "Sparse Matrix", "DiscretizedOperator", "Relative")
        @printf("%-40s | %-20s | %-20s | %-10s\n", "", "(time in μs)", "(time in μs)", "(DO/Sparse)")
        println("-" ^ 100)

        # Run benchmarks for each operation
        for op_info in operations
            op_name = op_info[1]
            sparse_op = op_info[2]
            dop_op = length(op_info) > 2 ? op_info[3] : nothing

            if op_name == "Matrix broadcast (@. A.matrix = B.matrix + B.matrix^2)"
                # Special case for direct matrix operation
                dop_result = @benchmark (@. $A_dop.matrix = $B_dop.matrix + $B_dop.matrix^2)
                dop_time = median(dop_result).time / 1000  # Convert ns to μs
                sparse_time = NaN
                relative = NaN
                @printf("%-40s | %-20s | %-20.3f | %-10s\n", op_name, "N/A", dop_time, "N/A")
                continue
            end

            if occursin("Chained operations", op_name) || occursin("Mixed operations", op_name)
                # Benchmark operations with three arguments
                sparse_result = @benchmark $sparse_op($A_sparse, $B_sparse, $C_sparse)
                sparse_time = median(sparse_result).time / 1000  # Convert ns to μs

                if dop_op !== nothing
                    dop_result = @benchmark $dop_op($A_dop, $B_dop, $C_dop)
                    dop_time = median(dop_result).time / 1000  # Convert ns to μs
                    relative = dop_time / sparse_time
                else
                    dop_time = NaN
                    relative = NaN
                end
            else
                # Benchmark operations with one or two arguments
                if applicable(sparse_op, A_sparse)
                    # Single argument operation
                    sparse_result = @benchmark $sparse_op($A_sparse)
                    sparse_time = median(sparse_result).time / 1000  # Convert ns to μs

                    if dop_op !== nothing
                        dop_result = @benchmark $dop_op($A_dop)
                        dop_time = median(dop_result).time / 1000  # Convert ns to μs
                        relative = dop_time / sparse_time
                    else
                        dop_time = NaN
                        relative = NaN
                    end
                elseif op_name == "Linear system solution (A\\b)"
                    # Special case for linear system solution
                    sparse_result = @benchmark $sparse_op($A_sparse, $b_vector)
                    sparse_time = median(sparse_result).time / 1000  # Convert ns to μs

                    if dop_op !== nothing
                        dop_result = @benchmark $dop_op($A_dop, $b_vector)
                        dop_time = median(dop_result).time / 1000  # Convert ns to μs
                        relative = dop_time / sparse_time
                    else
                        dop_time = NaN
                        relative = NaN
                    end
                else
                    # Two argument operation
                    sparse_result = @benchmark $sparse_op($A_sparse, $B_sparse)
                    sparse_time = median(sparse_result).time / 1000  # Convert ns to μs

                    if dop_op !== nothing
                        dop_result = @benchmark $dop_op($A_dop, $B_dop)
                        dop_time = median(dop_result).time / 1000  # Convert ns to μs
                        relative = dop_time / sparse_time
                    else
                        dop_time = NaN
                        relative = NaN
                    end
                end
            end

            # Print results
            @printf("%-40s | %-20.3f | %-20.3f | %-10.3f\n", op_name, sparse_time, dop_time, relative)
        end
    end
end

"""
    run_scaling_test(min_size, max_size, sparsity_level)

Run scaling tests to see how performance scales with matrix size.
"""
function run_scaling_test(min_size, max_size, num_points, sparsity_level)
    println("\nScaling Test")
    println("=============")

    # Generate logarithmically spaced matrix sizes
    sizes = round.(Int, exp.(range(log(min_size), log(max_size), length=num_points)))

    # Define operations to test for scaling
    operations = [
        ("Simple addition (A + B)",
            (A, B) -> A + B,
            (A, B) -> A + B),

        ("Element-wise multiplication (A .* B)",
            (A, B) -> A .* B,
            (A, B) -> A .* B),

        ("Complex broadcast (@. 2.0 * A + 0.5 * B)",
            (A, B) -> @. 2.0 * A + 0.5 * B,
            (A, B) -> @. 2.0 * A + 0.5 * B),

        ("Matrix inversion (inv(A))",
            A -> inv(A),
            A -> inv(A)),

        ("Linear system solution (A\\b)",
            (A, b) -> A \ b,
            (A, b) -> A \ b)
    ]

    # Set random seed for reproducibility
    rng = Random.MersenneTwister(42)

    # Run test for each operation
    for (op_name, sparse_op, dop_op) in operations
        println("\nOperation: $op_name")
        println("--------------------")

        @printf("%-15s | %-15s | %-15s | %-15s\n", "Matrix Size", "Sparse (μs)", "DiscretizedOp (μs)", "Ratio")
        println("-" ^ 70)

        for size in sizes
            # Generate matrices
            A_sparse = generate_test_sparse_matrix(size^2, sparsity_level, rng)
            B_sparse = generate_test_sparse_matrix(size^2, sparsity_level, rng)

            # Make sure matrices are well-conditioned for inversion
            A_sparse .+= sparse(one(Float64)*I, size^2, size^2)
            B_sparse .+= sparse(one(Float64)*I, size^2, size^2)

            # Generate random vector for solving linear systems
            b_vector = randn(rng, size^2)

            # Create DiscretizedOperator objects
            A_dop = create_discretized_operator(copy(A_sparse))
            B_dop = create_discretized_operator(copy(B_sparse))

            # Benchmark
            if op_name == "Linear system solution (A\\b)"
                sparse_result = @benchmark $sparse_op($A_sparse, $b_vector)
                dop_result = @benchmark $dop_op($A_dop, $b_vector)
            elseif op_name == "Matrix inversion (inv(A))"
                sparse_result = @benchmark $sparse_op($A_sparse)
                dop_result = @benchmark $dop_op($A_dop)
            else
                sparse_result = @benchmark $sparse_op($A_sparse, $B_sparse)
                dop_result = @benchmark $dop_op($A_dop, $B_dop)
            end

            # Calculate times
            sparse_time = median(sparse_result).time / 1000  # Convert ns to μs
            dop_time = median(dop_result).time / 1000  # Convert ns to μs
            relative = dop_time / sparse_time

            @printf("%-15d | %-15.3f | %-15.3f | %-15.3f\n", size, sparse_time, dop_time, relative)
        end
    end
end

# Main benchmark function
function main()
    println("# Performance Comparison: SparseMatrixCSC vs DiscretizedOperator")
    println("================================================================")

    # Run detailed benchmarks for specific sizes
    sizes = [20, 50, 100]  # These give 400x400, 2500x2500, 10000x10000 matrices
    sparsity = 0.01        # 1% non-zero elements
    run_benchmarks(sizes, sparsity)

    # Run scaling test
    min_size = 10          # 100x100 matrix
    max_size = 200         # 40000x40000 matrix
    num_points = 5
    run_scaling_test(min_size, max_size, num_points, sparsity)
end

# Run the benchmark if this script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end