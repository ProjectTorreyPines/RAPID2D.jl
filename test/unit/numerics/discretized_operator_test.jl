# Test suite for DiscretizedOperator arithmetic operations
using Test
using RAPID2D
using RAPID2D.SparseArrays
using RAPID2D.LinearAlgebra

@testset "DiscretizedOperator" begin
    # Test setup: Create a small test grid and basic operators
    @testset "Constructor" begin
        # Test basic constructor
        dims_rz = (5, 5)
        I = [1, 2, 3]
        J = [1, 2, 3]
        V = [1.0, 2.0, 3.0]

        dop = DiscretizedOperator(dims_rz, I, J, V)

        @test dop.dims_rz == dims_rz
        @test dop.matrix isa SparseMatrixCSC{Float64}
        @test length(dop.k2csc) == 3
        @test size(dop.matrix) == (25, 25) # 5*5 = 25 elements in total
        @test dop.matrix[1, 1] == 1.0
        @test dop.matrix[2, 2] == 2.0
        @test dop.matrix[3, 3] == 3.0

        # Test the empty constructor
        empty_dop = DiscretizedOperator{Float64}(dims_rz)
        @test empty_dop.dims_rz == dims_rz
        @test isempty(empty_dop.k2csc)
        @test size(empty_dop.matrix) == (25, 25)
        @test nnz(empty_dop.matrix) == 0
    end

    # Helper function to create test operators
    function create_test_operators()
        # Create a simple 3x3 grid for testing
        NR, NZ = 3, 3
        dims_rz = (NR, NZ)

        # Identity-like operator
        I_id = Int[]
        J_id = Int[]
        V_id = Float64[]

        # First derivative in R direction (central difference)
        I_dr = Int[]
        J_dr = Int[]
        V_dr = Float64[]

        # Fill the identity operator
        for i in 1:NR*NZ
            push!(I_id, i)
            push!(J_id, i)
            push!(V_id, 1.0)
        end

        # Fill a central difference operator for dR
        # (only interior points, boundary points are zero)
        k = 1
        for j in 1:NZ
            for i in 1:NR
                if i > 1 && i < NR
                    # East neighbor
                    push!(I_dr, k)
                    push!(J_dr, k+1)
                    push!(V_dr, 0.5)

                    # West neighbor
                    push!(I_dr, k)
                    push!(J_dr, k-1)
                    push!(V_dr, -0.5)
                end
                k += 1
            end
        end

        # Create the operators
        id_op = DiscretizedOperator(dims_rz, I_id, J_id, V_id)
        dr_op = DiscretizedOperator(dims_rz, I_dr, J_dr, V_dr)

        return id_op, dr_op
    end

    @testset "Basic Operations" begin
        id_op, dr_op = create_test_operators()

        # Test equality operator
        @test id_op == id_op
        @test dr_op == dr_op
        @test id_op != dr_op

        # Test multiplication with vector
        v = ones(9)
        result = id_op * v
        @test result ≈ v

        # Test multiplication with matrix
        m = ones(3, 3)
        result_mat = id_op * m
        @test result_mat ≈ m

        # Test that dR operator zeros out constant field
        const_vec = ones(9)
        @test norm(dr_op * const_vec) ≈ 0 atol=1e-10

        # Test dR operator on a linear field
        linear_field = reshape(1.0:9.0, 3, 3)
        linear_vec = reshape(linear_field, 9)

        # The result should have non-zero values for interior points
        dr_result = dr_op * linear_vec
        # Check that central difference for interior points is correct
        # For our simple case, the derivative of the linear field should be constant 1.0
        # in interior points
        @test dr_result[2] ≈ 1.0 atol=1e-10
        @test dr_result[5] ≈ 1.0 atol=1e-10
        @test dr_result[8] ≈ 1.0 atol=1e-10
    end

    @testset "Arithmetic Operations" begin
        id_op, dr_op = create_test_operators()

        @testset "Unary Minus" begin
            neg_id_op = -id_op
            @test neg_id_op.dims_rz == id_op.dims_rz
            @test all(neg_id_op.matrix .≈ -id_op.matrix)

            # Test with vector
            v = ones(9)
            @test (-id_op * v) ≈ -(id_op * v)
        end

        @testset "Scalar Multiplication" begin
            # Test left multiplication
            scaled_op = 2.0 * id_op
            @test scaled_op.dims_rz == id_op.dims_rz
            @test all(scaled_op.matrix.nzval .≈ 2.0 * id_op.matrix.nzval)

            # Test right multiplication
            scaled_op2 = id_op * 2.0
            @test scaled_op2 == scaled_op

            # Test with vector
            v = ones(9)
            @test (2.0 * id_op * v) ≈ 2.0 * (id_op * v)
        end

        @testset "Scalar Division" begin
            scaled_op = id_op / 2.0
            @test scaled_op.dims_rz == id_op.dims_rz
            @test all(scaled_op.matrix.nzval .≈ id_op.matrix.nzval / 2.0)

            # Test with vector
            v = ones(9)
            @test (id_op / 2.0 * v) ≈ (id_op * v) / 2.0
        end

        @testset "Element-wise Scalar Operations" begin
            # Element-wise multiplication
            scaled_op = 2.0 .* id_op
            @test scaled_op.dims_rz == id_op.dims_rz
            @test all(scaled_op.matrix.nzval .≈ 2.0 .* id_op.matrix.nzval)

            # Element-wise division
            scaled_op = id_op ./ 2.0
            @test scaled_op.dims_rz == id_op.dims_rz
            @test all(scaled_op.matrix.nzval .≈ id_op.matrix.nzval ./ 2.0)

            # Element-wise division (scalar by operator)
            inv_op = 1.0 ./ id_op
            @test inv_op.dims_rz == id_op.dims_rz
            # Only check non-zeros to avoid division by zero
            for (i, j, v) in zip(findnz(id_op.matrix)...)
                @test inv_op.matrix[i, j] ≈ 1.0 / v
            end
        end

        @testset "Addition and Subtraction" begin
            # Addition
            sum_op = id_op + dr_op
            @test sum_op.dims_rz == id_op.dims_rz
            @test all(sum_op.matrix .≈ id_op.matrix + dr_op.matrix)

            # Subtraction
            diff_op = id_op - dr_op
            @test diff_op.dims_rz == id_op.dims_rz
            @test all(diff_op.matrix .≈ id_op.matrix - dr_op.matrix)

            # Test dimension mismatch
            wrong_dims_rz = DiscretizedOperator{Float64}((4, 4))
            @test_throws DimensionMismatch id_op + wrong_dims_rz
            @test_throws DimensionMismatch id_op - wrong_dims_rz
        end

        @testset "Element-wise Operations Between Operators" begin
            # Element-wise multiplication
            prod_op = id_op .* dr_op
            @test prod_op.dims_rz == id_op.dims_rz
            @test all(prod_op.matrix .≈ id_op.matrix .* dr_op.matrix)

            # Element-wise division
            div_op = id_op ./ (id_op .+ 0.1)  # Adding 0.1 to avoid division by zero
            @test div_op.dims_rz == id_op.dims_rz
            @test all(div_op.matrix .≈ id_op.matrix ./ (id_op.matrix .+ 0.1))

            # Test dimension mismatch
            wrong_dims_rz = DiscretizedOperator{Float64}((4, 4))
            @test_throws DimensionMismatch id_op .* wrong_dims_rz
            @test_throws DimensionMismatch id_op ./ wrong_dims_rz
        end

        @testset "Operator Composition" begin
            # Composing an operator with itself
            id_squared = id_op * id_op
            @test id_squared.dims_rz == id_op.dims_rz
            @test all(id_squared.matrix .≈ id_op.matrix * id_op.matrix)

            # Composing different operators
            dr_id = dr_op * id_op
            @test dr_id.dims_rz == id_op.dims_rz
            @test all(dr_id.matrix .≈ dr_op.matrix * id_op.matrix)

            # Test dimensions
            wrong_dims_rz = DiscretizedOperator{Float64}((4, 4))
            @test_throws DimensionMismatch id_op * wrong_dims_rz
        end
    end

    @testset "Complex Operations" begin
        id_op, dr_op = create_test_operators()

        # Test a complex expression combining multiple operations
        complex_op = 2.0 * id_op - 0.5 * dr_op
        @test complex_op.dims_rz == id_op.dims_rz
        @test all(complex_op.matrix .≈ 2.0 * id_op.matrix - 0.5 * dr_op.matrix)

        # Test operator composition with scaling
        composed_op = (2.0 * id_op) * (0.5 * dr_op)
        @test composed_op.dims_rz == id_op.dims_rz
        @test all(composed_op.matrix .≈ (2.0 * id_op.matrix) * (0.5 * dr_op.matrix))

        # Test associativity of scalar multiplication
        op1 = 2.0 * (3.0 * id_op)
        op2 = (2.0 * 3.0) * id_op
        @test op1 == op2
    end

    @testset "Fused Operations with @. Macro" begin
        id_op, dr_op = create_test_operators()

        # Test fused operations with the @. macro
        @testset "Element-wise Multiplication with @." begin
            # Create a copy for testing
            op1 = id_op
            op2 = dr_op

            # Test element-wise multiplication with @. macro
            result1 = @. op1 * op2
            # Compare with manually doing the operation
            result2 = op1 .* op2
            @test result1 == result2

            # Verify the result has correct dimensions and properties
            @test result1.dims_rz == op1.dims_rz
            # More efficient sparse matrix comparison
            @test result1.matrix == op1.matrix .* op2.matrix
        end

        @testset "Element-wise Division with @." begin
            # Avoid division by zero by adding a small value
            op1 = id_op
            op2 = id_op .+ 0.1

            # Test element-wise division with @. macro
            result1 = @. op1 / op2
            # Compare with manually doing the operation
            result2 = op1 ./ op2
            @test result1 == result2

            # Verify the result has correct dimensions and properties
            @test result1.dims_rz == op1.dims_rz
            # More efficient sparse matrix comparison
            @test result1.matrix == op1.matrix ./ op2.matrix
        end

        @testset "Complex Fused Operations with @." begin
            # Test more complex fused operations
            op1 = id_op
            op2 = dr_op
            α = 2.0
            β = 0.5

            # Fused operation using @. macro
            result1 = @. α * op1 + β * op2

            # Manually construct the expected result
            result2 = α * op1 + β * op2

            @test result1 == result2
            @test result1.dims_rz == op1.dims_rz
            # More efficient sparse matrix comparison
            @test result1.matrix == α * op1.matrix + β * op2.matrix
        end

        @testset "Fused Operation with Scalar and Operator" begin
            op = id_op
            α = 2.0

            # Test fused operation with scalar and operator
            result1 = @. α * op + 1

            # Manually construct the expected result
            result2 = α * op .+ 1

            @test result1.dims_rz == op.dims_rz
            # More efficient sparse matrix comparison
            @test result1.matrix == α * op.matrix .+ 1
        end

        @testset "Verify Matrix Equivalence" begin
            op1 = id_op
            op2 = dr_op

            # Compare a regular operation with its fused equivalent
            regular_result = 2.0 * op1 + 0.5 * op2
            fused_result = @. 2.0 * op1 + 0.5 * op2

            @test regular_result == fused_result
            # More efficient sparse matrix comparison
            @test regular_result.matrix == fused_result.matrix

            # Test chained operations with and without fusion
            chainA = (2.0 * op1) .* op2
            chainB = @. (2.0 * op1) * op2

            @test chainA.dims_rz == chainB.dims_rz
            # More efficient sparse matrix comparison
            @test chainA.matrix == chainB.matrix
        end
    end
end