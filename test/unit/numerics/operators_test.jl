# Discretized differential operators: ∂R, ∂Z, 𝐽⁻¹∂R𝐽 and the divergence.
# Each testitem builds and mutates its OWN RAPID object; the shared
# SimulationConfig factories live in setup_numerics.jl.

@testitem "Basic differential operators" setup = [NumericsFixtures] begin
    # Define test parameters
    NR, NZ = 50, 100  # Small grid for testing
    FT = Float64    # Floating point type

    # Create a proper RAPID object for testing using create_rapid_object
    config = gas_filled_config(FT; NR = NR, NZ = NZ)
    # Create the RAPID object
    RP = create_rapid_object(; config = config)


    FR = zeros(FT, NR, NZ)
    FZ = zeros(FT, NR, NZ)
    # Create a vortex-like flow vector field
    for j in 1:NZ, i in 1:NR
        # Centered coordinates
        x = i - NR / 2
        y = j - NZ / 2
        r = sqrt(x^2 + y^2)

        if r > 0
            # Circular flow around center
            FR[i, j] = -0.1 * y / r
            FZ[i, j] = 0.1 * x / r
        else
            FR[i, j] = 0.0
            FZ[i, j] = 0.0
        end
    end

    OP = RP.operators

    # Explicit method
    div_numerical_1 = calculate_divergence(RP.G, FR, FZ) # Using central differencing

    # using operators matrix-vector multiplication
    div_numerical_2 = OP.𝐽⁻¹∂R_𝐽 * FR .+ OP.∂Z * FZ

    @test isapprox(div_numerical_1, div_numerical_2, rtol = 1.0e-14)

    # Test convient dispatches
    @test div_numerical_2 == reshape(calculate_divergence(OP, FR[:], FZ[:]), NR, NZ)
    @test div_numerical_2 == calculate_divergence(OP, FR, FZ)

    @testset "Convenient dispatches" begin
        ∂R = RAPID2D.construct_∂R_operator(RP)
        𝐽⁻¹∂R_𝐽 = construct_𝐽⁻¹∂R_𝐽_operator(RP)
        ∂Z = construct_∂Z_operator(RP)

        @test ∂R == OP.∂R
        @test 𝐽⁻¹∂R_𝐽 == OP.𝐽⁻¹∂R_𝐽
        @test ∂Z == OP.∂Z
    end

    # Analytical test cases
    @testset "Analytical divergence tests" begin
        # Get grid coordinates
        R2D = RP.G.R2D
        Z2D = RP.G.Z2D
        interior_points = 2:(NR - 1), 2:(NZ - 1)

        # Test Case 1: Radial Flow (FR = R, FZ = 0)
        let
            k = 1.0  # Constant
            FR = k .* R2D
            FZ = zeros(FT, NR, NZ)

            div_numerical = RAPID2D.calculate_divergence(RP.G, FR, FZ)  # Using central differencing
            div_analytical = fill(2 * k, (NR, NZ))

            @test isapprox(div_numerical[interior_points...], div_analytical[interior_points...], rtol = 1.0e-4)
        end

        # Test Case 2: Linear Flow (FR = a·R, FZ = b·Z)
        let
            a = 1.0
            b = 2.0
            FR = a .* R2D
            FZ = b .* Z2D

            div_numerical = RAPID2D.calculate_divergence(RP.G, FR, FZ)  # Using central differencing
            div_analytical = fill(2 * a + b, (NR, NZ))

            @test isapprox(div_numerical[interior_points...], div_analytical[interior_points...], rtol = 1.0e-2)
        end

        # Test Case 3 (divergence-free): (FR = R²-Z², FZ = -3*R*Z + Z³/(3*R))
        let
            FR = @. R2D^2 - Z2D^2
            FZ = @. -3 * R2D * Z2D + Z2D^3 / (3 * R2D)
            div_numerical = RAPID2D.calculate_divergence(RP.G, FR, FZ)  # Using central differencing

            @test maximum(abs.(div_numerical[interior_points...])) < 1.0e-2
        end
    end
end

@testitem "no whole-grid transport operators: nothing is cached, nothing sweeps the band" begin
    # The diffusion, advection and convective-divergence builders swept 2:N-1 with no wall
    # awareness and were refreshed once per step into `Operators`. All transport operators
    # are now built on in-wall rows where they are used (wall_diffusion.jl, face_flux.jl,
    # primitive_advection.jl); nothing of the old family may survive as a silent fallback.
    for name in (
            :compute_∇𝐃∇f_directly, :construct_∇𝐃∇_operator, :update_∇𝐃∇_operator!,
            :compute_𝐮∇f_directly, :construct_𝐮∇_operator, :update_𝐮∇_operator!,
            :compute_∇f𝐮_directly, :construct_∇𝐮_operator, :update_∇𝐮_operator!,
            :update_transport_related_operators!,
        )
        @test !isdefined(RAPID2D, name)
    end
    for field in (:∇𝐃∇, :𝐮∇, :∇𝐮)
        @test !(field in fieldnames(RAPID2D.Operators{Float64}))
    end
end
