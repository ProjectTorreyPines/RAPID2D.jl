"""
Test suite for magnetic field calculations from poloidal flux function ψ.

Tests the calculate_B_from_ψ! and calculate_B_from_ψ functions using analytical solutions.
"""

using Test
using RAPID2D

@testset "Magnetic Field Calculations from ψ" begin

    # Test parameters
    FT = Float64

	NR, NZ = 40, 80
	G = GridGeometry{FT}(40, 80)
	initialize_grid_geometry!(G, (0.5, 1.5), (-1.0, 1.0))

    @testset "Uniform field test (ψ = constant)" begin
        # Test case 1: ψ = constant → BR = BZ = 0
        ψ = ones(FT, NR, NZ) * 5.0  # constant flux
        BR = zeros(FT, NR, NZ)
        BZ = zeros(FT, NR, NZ)

        calculate_B_from_ψ!(RP.G, ψ, BR, BZ)

        @test all(abs.(BR) .< 1e-14)
        @test all(abs.(BZ) .< 1e-14)
    end

    @testset "Mixed field test (ψ = aR² + bZ²)" begin
        # Test case 4: ψ = a*R² + b*Z² → BR = -2b*Z/R, BZ = 2a
        a, b = 1.0, 0.8
		ψ = @. a * G.R2D^2 + b * G.Z2D^2

        BR, BZ = calculate_B_from_ψ(G, ψ)

		expected_BR = @. -2*b * G.Z2D / G.R2D
		expected_BZ = 2 *a * ones(FT, NR, NZ)

		# Check interior points (higher accuracy)
		interiors = (2:NR-1, 2:NZ-1)
		@test isapprox(BR[interiors...], expected_BR[interiors...], rtol=1e-10)
		@test isapprox(BZ[interiors...], expected_BZ[interiors...], rtol=1e-10)

		# Check including boundaries (lower accuracy)
		@test isapprox(BR, expected_BR, rtol=1e-2)
		@test isapprox(BZ, expected_BZ, rtol=1e-2)
    end

    @testset "Wrapper function test" begin
        # Test the non-mutating wrapper function
        ψ = ones(FT, NR, NZ) * 3.0  # constant flux

        BR_result, BZ_result = calculate_B_from_ψ(G, ψ)

        @test size(BR_result) == (NR, NZ)
        @test size(BZ_result) == (NR, NZ)
        @test all(abs.(BR_result) .< 1e-14)
        @test all(abs.(BZ_result) .< 1e-14)
    end

    @testset "Error handling tests" begin
        # Test dimension mismatches
        ψ_wrong = ones(FT, NR+1, NZ)
        BR = zeros(FT, NR, NZ)
        BZ = zeros(FT, NR, NZ)

        @test_throws AssertionError calculate_B_from_ψ!(G, ψ_wrong, BR, BZ)

        ψ = ones(FT, NR, NZ)
        BR_wrong = zeros(FT, NR, NZ+1)

        @test_throws AssertionError calculate_B_from_ψ!(G, ψ, BR_wrong, BZ)
    end
end
