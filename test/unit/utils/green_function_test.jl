# Green's function implementation.

@testitem "Green's Function" begin

    @testset "Basic functionality" begin
        # Simple test case
        R_dest = [1.0, 1.5]
        Z_dest = [0.0, 0.5]
        R_src = [2.0]
        Z_src = [0.0]
        I_src = [1.0e6]  # 1 MA current

        # Test without derivatives
        ψ = calculate_ψ_by_green_function(R_dest, Z_dest, R_src, Z_src, I_src)

        @test size(ψ) == (2, 1)
        @test all(isfinite.(ψ))
        @test !any(isnan.(ψ))

        # Test with derivatives
        ψ_with_deriv, derivatives = calculate_ψ_by_green_function(
            R_dest, Z_dest, R_src, Z_src, I_src; compute_derivatives = true
        )

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
            [1.0], [0.0, 0.5], [2.0], [0.0], [1.0e6]
        )

        # Test mismatched source coordinates
        @test_throws AssertionError calculate_ψ_by_green_function(
            [1.0], [0.0], [2.0], [0.0, 0.5], [1.0e6]
        )

        # Test invalid current array size
        @test_throws AssertionError calculate_ψ_by_green_function(
            [1.0], [0.0], [2.0, 2.5], [0.0, 0.5], [1.0e6, 2.0e6, 3.0e6]
        )
    end

    @testset "Scalar current" begin
        # Test with scalar current applied to multiple sources
        R_dest = [1.0]
        Z_dest = [0.0]
        R_src = [2.0, 2.5]
        Z_src = [0.0, 0.5]
        I_src = 1.0e6  # Scalar current

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

        I1 = [1.0e6]
        I2 = [2.0e6]

        ψ1 = calculate_ψ_by_green_function(R_dest, Z_dest, R_src, Z_src, I1)
        ψ2 = calculate_ψ_by_green_function(R_dest, Z_dest, R_src, Z_src, I2)

        @test ψ2[1] ≈ 2.0 * ψ1[1] rtol = 1.0e-12

        # Test symmetry: ψ should be the same if we swap source and destination
        # (for equal R values due to the Green's function symmetry)
        R_dest2 = [2.0]
        Z_dest2 = [0.0]
        R_src2 = [1.0]
        Z_src2 = [0.0]

        ψ_swapped = calculate_ψ_by_green_function(R_dest2, Z_dest2, R_src2, Z_src2, I1)

        # The Green's function is symmetric in the sense that G(r,r') = G(r',r)
        @test ψ1[1] ≈ ψ_swapped[1] rtol = 1.0e-12
    end

    @testset "Multiple source-destination points" begin
        # Test with multiple source and destination points
        R_dest = [1.0 1.5; 2.0 1.2]
        Z_dest = [0.0 0.5; 1.0 -0.5]
        R_src = [2.0, 2.5, 3.0]
        Z_src = [0.0, 0.5, 2.0]
        I_src = [1.0e6, 2.0e6, 1.0]

        ψ = RAPID2D.calculate_ψ_by_green_function(R_dest, Z_dest, R_src, Z_src, I_src)
        @test size(ψ) == (2, 2, 3)
        @test all(isfinite.(ψ))

        # Test with derivatives
        ψ_with_deriv, derivatives = calculate_ψ_by_green_function(
            R_dest, Z_dest, R_src, Z_src, I_src; compute_derivatives = true
        )

        @test ψ ≈ ψ_with_deriv
        @test all(isfinite.(derivatives.dψ_dRdest))
        @test all(isfinite.(derivatives.dψ_dZdest))
        @test all(isfinite.(derivatives.dψ_dRsrc))
        @test all(isfinite.(derivatives.dψ_dZsrc))
    end
end

@testitem "Green's function: derivatives match finite differences" begin
    using RAPID2D: calculate_ψ_by_green_function
    # ψ = 2e-7 I √(R_d R_s/m) f(m): the R-derivatives carry √(R_d R_s) as well as m. A 2×3 array
    # of destinations and four sources with currents other than one, near and far, on both sides
    # in R; no destination sits on a source.
    R_dest = [1.1 1.45 2.3; 1.6 1.95 0.85]
    Z_dest = [0.05 -0.4 0.3; 0.9 -0.75 0.1]
    R_src = [1.2, 1.5, 2.1, 0.95]
    Z_src = [0.2, -0.1, 0.6, -0.5]
    I_src = [1.0, -2.5, 0.7, 3.2]
    _, d = calculate_ψ_by_green_function(R_dest, Z_dest, R_src, Z_src, I_src; compute_derivatives = true)

    h = 1.0e-6
    ψ_at(Rd, Zd, Rs, Zs) = calculate_ψ_by_green_function(Rd, Zd, Rs, Zs, I_src)
    fd_Rdest = (ψ_at(R_dest .+ h, Z_dest, R_src, Z_src) .- ψ_at(R_dest .- h, Z_dest, R_src, Z_src)) ./ 2h
    fd_Zdest = (ψ_at(R_dest, Z_dest .+ h, R_src, Z_src) .- ψ_at(R_dest, Z_dest .- h, R_src, Z_src)) ./ 2h
    fd_Rsrc = (ψ_at(R_dest, Z_dest, R_src .+ h, Z_src) .- ψ_at(R_dest, Z_dest, R_src .- h, Z_src)) ./ 2h
    fd_Zsrc = (ψ_at(R_dest, Z_dest, R_src, Z_src .+ h) .- ψ_at(R_dest, Z_dest, R_src, Z_src .- h)) ./ 2h

    @test d.dψ_dRdest ≈ fd_Rdest rtol = 1.0e-6
    @test d.dψ_dZdest ≈ fd_Zdest rtol = 1.0e-6
    @test d.dψ_dRsrc ≈ fd_Rsrc rtol = 1.0e-6
    @test d.dψ_dZsrc ≈ fd_Zsrc rtol = 1.0e-6
end
