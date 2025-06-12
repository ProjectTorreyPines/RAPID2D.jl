"""
Test script for Green's function implementation
"""

using RAPID2D
using Test
using RAPID2D.SpecialFunctions

@testset "Green's Function Tests" begin

    @testset "Basic functionality" begin
        # Simple test case
        R_dest = [1.0, 1.5]
        Z_dest = [0.0, 0.5]
        R_src = [2.0]
        Z_src = [0.0]
        I_src = [1e6]  # 1 MA current

        # Test without derivatives
        ψ = calculate_ψ_by_green_function(R_dest, Z_dest, R_src, Z_src, I_src)

        @test size(ψ) == (2, 1)
        @test all(isfinite.(ψ))
        @test !any(isnan.(ψ))

        # Test with derivatives
        ψ_with_deriv, derivatives = calculate_ψ_by_green_function(
            R_dest, Z_dest, R_src, Z_src, I_src; compute_derivatives=true)

        @test ψ ≈ ψ_with_deriv
        @test haskey(derivatives, :dψ_dRdest)
        @test haskey(derivatives, :dψ_dZdest)
        @test haskey(derivatives, :dψ_dRsrc)
        @test haskey(derivatives, :dψ_dZsrc)

        # Check derivative dimensions
        @test size(derivatives.dψ_dRdest) == size(ψ)
        @test size(derivatives.dψ_dZdest) == size(ψ)
        @test size(derivatives.dψ_dRsrc) == size(ψ)
        @test size(derivatives.dψ_dZsrc) == size(ψ)
    end

    @testset "Input validation" begin
        # Test mismatched destination coordinates
        @test_throws AssertionError calculate_ψ_by_green_function(
            [1.0], [0.0, 0.5], [2.0], [0.0], [1e6])

        # Test mismatched source coordinates
        @test_throws AssertionError calculate_ψ_by_green_function(
            [1.0], [0.0], [2.0], [0.0, 0.5], [1e6])

        # Test invalid current array size
        @test_throws AssertionError calculate_ψ_by_green_function(
            [1.0], [0.0], [2.0, 2.5], [0.0, 0.5], [1e6, 2e6, 3e6])
    end

    @testset "Scalar current" begin
        # Test with scalar current applied to multiple sources
        R_dest = [1.0]
        Z_dest = [0.0]
        R_src = [2.0, 2.5]
        Z_src = [0.0, 0.5]
        I_src = 1e6  # Scalar current

        ψ = calculate_ψ_by_green_function(R_dest, Z_dest, R_src, Z_src, I_src)
        @test size(ψ) == (1, 2)
        @test all(isfinite.(ψ))
    end

    @testset "Physical properties" begin
        # Test that ψ scales linearly with current
        R_dest = [1.0]
        Z_dest = [0.0]
        R_src = [2.0]
        Z_src = [0.0]

        I1 = [1e6]
        I2 = [2e6]

        ψ1 = calculate_ψ_by_green_function(R_dest, Z_dest, R_src, Z_src, I1)
        ψ2 = calculate_ψ_by_green_function(R_dest, Z_dest, R_src, Z_src, I2)

        @test ψ2[1] ≈ 2.0 * ψ1[1] rtol=1e-12

        # Test symmetry: ψ should be the same if we swap source and destination
        # (for equal R values due to the Green's function symmetry)
        R_dest2 = [2.0]
        Z_dest2 = [0.0]
        R_src2 = [1.0]
        Z_src2 = [0.0]

        ψ_swapped = calculate_ψ_by_green_function(R_dest2, Z_dest2, R_src2, Z_src2, I1)

        # The Green's function is symmetric in the sense that G(r,r') = G(r',r)
        @test ψ1[1] ≈ ψ_swapped[1] rtol=1e-12
    end

    @testset "Multiple source-destination points" begin
        # Test with multiple source and destination points
        R_dest = [1.0 1.5; 2.0 1.2]
        Z_dest = [0.0 0.5; 1.0 -0.5]
        R_src = [2.0, 2.5, 3.0]
        Z_src = [0.0, 0.5, 2.0]
        I_src = [1e6, 2e6, 1.0]

        ψ = RAPID2D.calculate_ψ_by_green_function(R_dest, Z_dest, R_src, Z_src, I_src)
        @test size(ψ) == (2, 2, 3)
        @test all(isfinite.(ψ))

        # Test with derivatives
        ψ_with_deriv, derivatives = calculate_ψ_by_green_function(
            R_dest, Z_dest, R_src, Z_src, I_src; compute_derivatives=true)

        @test ψ ≈ ψ_with_deriv
        @test all(isfinite.(derivatives.dψ_dRdest))
        @test all(isfinite.(derivatives.dψ_dZdest))
        @test all(isfinite.(derivatives.dψ_dRsrc))
        @test all(isfinite.(derivatives.dψ_dZsrc))
    end
end

